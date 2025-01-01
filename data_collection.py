
#%% collect the constituent stocks of EFT 0050.TW with each financial statement period.
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
dict0={'成分股':df0} # '成分股' -> constituent stocks
#932 stocks

#%% basic of crawl
url='https://goodinfo.tw/tw/StockFinCompare.asp?STOCK0=台積電&STOCK1=聯電&STOCK2=世界&STOCK3=旺宏&STOCK4=&RPT_CAT=XX_QUAR&RPT_TYPE=NM&selYEAR=2020&selQUAR=4&btnQry=%C2%A0查%C2%A0%C2%A0%C2%A0詢%C2%A0'
headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15'}
#headers -> 每台電腦不同
res=requests.get(url,headers=headers)
res.encoding='utf-8'
soup=BeautifulSoup(res.text,'lxml')
data=soup.select_one('#divDetail')
l=pd.read_html(data.prettify())
df=l[0]

#%% 爬goodinfo資料
# get financial statement data through the website 'goodinfo'

#個別財務比率表（單季）
#https://goodinfo.tw/tw/StockFinCompare.asp?STOCK0=台積電&STOCK1=聯電&STOCK2=世界&STOCK3=旺宏&STOCK4=&RPT_CAT=XX_QUAR&RPT_TYPE=NM&selYEAR=2020&selQUAR=4&btnQry=%C2%A0查%C2%A0%C2%A0%C2%A0詢%C2%A0
def crawl_financial_ratio_table(stock1,stock2,stock3,stock4,stock5,year,quar):
    url=('https://goodinfo.tw/StockInfo/StockFinCompare.asp?'
         +'STOCK0='+stock1
         +'&STOCK1='+stock2
         +'&STOCK2='+stock3
         +'&STOCK3='+stock4
         +'&STOCK4='+stock5
         +'&RPT_CAT=XX_QUAR&RPT_TYPE=NM'
         +'&selYEAR='+year
         +'&selQUAR='+quar
         +'&btnQry=%C2%A0查%C2%A0%C2%A0%C2%A0詢%C2%A0')
    headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15'}
    res=requests.get(url,headers=headers)
    res.encoding='utf-8'
    soup=BeautifulSoup(res.text,'lxml')
    data=soup.select_one('#divDetail')
    l=pd.read_html(data.prettify())
    df=l[0]
    
    return df

#個別損益表（單季）
#https://goodinfo.tw/StockInfo/StockFinCompare.asp?STOCK0=台積電&STOCK1=聯電&STOCK2=世界&STOCK3=旺宏&STOCK4=&RPT_CAT=IS_QUAR&RPT_TYPE=NM&selYEAR=2020&selQUAR=4&btnQry=%C2%A0查%C2%A0%C2%A0%C2%A0詢%C2%A0
def crawl_income_statement(stock1,stock2,stock3,stock4,stock5,year,quar):
    url=('https://goodinfo.tw/StockInfo/StockFinCompare.asp?'
         +'STOCK0='+stock1
         +'&STOCK1='+stock2
         +'&STOCK2='+stock3
         +'&STOCK3='+stock4
         +'&STOCK4='+stock5
         +'&RPT_CAT=IS_QUAR&RPT_TYPE=NM'
         +'&selYEAR='+year
         +'&selQUAR='+quar
         +'&btnQry=%C2%A0查%C2%A0%C2%A0%C2%A0詢%C2%A0')
    headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15'}
    res=requests.get(url,headers=headers)
    res.encoding='utf-8'
    soup=BeautifulSoup(res.text,'lxml')
    data=soup.select_one('#divDetail')
    l=pd.read_html(data.prettify())
    df=l[0]
    
    return df

#現金流量表（單季）
#https://goodinfo.tw/StockInfo/StockFinCompare.asp?STOCK0=中華電&STOCK1=富邦金&STOCK2=聯電&STOCK3=國泰金&STOCK4=&RPT_CAT=CF_QUAR&RPT_TYPE=NM&selYEAR=2010&selQUAR=1&btnQry=%C2%A0查%C2%A0%C2%A0%C2%A0詢%C2%A0
def crawl_cashflow_statement(stock1,stock2,stock3,stock4,stock5,year,quar):
    url=('https://goodinfo.tw/StockInfo/StockFinCompare.asp?'
         +'STOCK0='+stock1
         +'&STOCK1='+stock2
         +'&STOCK2='+stock3
         +'&STOCK3='+stock4
         +'&STOCK4='+stock5
         +'&RPT_CAT=CF_QUAR&RPT_TYPE=NM'
         +'&selYEAR='+year
         +'&selQUAR='+quar
         +'&btnQry=%C2%A0查%C2%A0%C2%A0%C2%A0詢%C2%A0')
    headers={'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15'}
    res=requests.get(url,headers=headers)
    res.encoding='utf-8'
    soup=BeautifulSoup(res.text,'lxml')
    data=soup.select_one('#divDetail')
    l=pd.read_html(data.prettify())
    df=l[0]
    
    return df

#%% 個別財務比率表（單季）
import time
l_financial_ratio_table=['營業利益率','營業毛利率','股東權益報酬率  (當季)',
                         '資產報酬率  (當季)','稅前淨利率','營業利益季成長率',
                         '毛利季成長率','稅前淨利季成長率','稅後淨利季成長率',
                         '營收季成長率','資產總額季成長率',
                         '總資產週轉率 (次/年)','固定資產週轉率 (次/年)',
                         '存貨週轉率 (次/年)',
                         '流動比','速動比','負債總額 (%)','長期資金適合率']

n=len(dict0['成分股']['code'])


for year in range(2013,2014):
    for quar in range(4,4+1):
        year,quar=str(year),str(quar)
        df0=pd.DataFrame()
        for i in range(0,n,5):
            stock1=str(dict0['成分股']['code'][i])
            stock2=str(dict0['成分股']['code'][i+1])
            stock3=str(dict0['成分股']['code'][i+2])
            stock4=str(dict0['成分股']['code'][i+3])
            stock5=str(dict0['成分股']['code'][i+4])
            
            df=crawl_financial_ratio_table(stock1,stock2,stock3,stock4,stock5,year,quar)
            df.set_index(df.columns[0],inplace=True)
            #df=df.loc[l_financial_ratio_table,:]
            df=df.loc[[j for j in df.index if j in l_financial_ratio_table],:]
            #df.drop('-',axis=1,inplace=True)
            df0=pd.concat([df0,df],axis=1)
    
            time.sleep(30)
        dict0['個別財務比率表（單季）'+year+'_'+quar]=df0

#2013_4 error
#2014_1

#%% 現金流量表（單季）

for year in range(2020,2021):
    for quar in range(1,4+1):
        year,quar=str(year),str(quar)
        df0=pd.DataFrame()
        for i in range(0,n,5):
            stock1=str(dict0['成分股']['code'][i])
            stock2=str(dict0['成分股']['code'][i+1])
            stock3=str(dict0['成分股']['code'][i+2])
            stock4=str(dict0['成分股']['code'][i+3])
            stock5=str(dict0['成分股']['code'][i+4])
            
            df=crawl_cashflow_statement(stock1,stock2,stock3,stock4,stock5,year,quar)
            df.set_index(df.columns[0],inplace=True)
            #df.drop('-',axis=1,inplace=True)
            df=df.loc['本期淨利(淨損)',:]
            df=pd.DataFrame(df).T
            df0=pd.concat([df0,df],axis=1)
    
            time.sleep(25)
        dict0['現金流量表（單季）'+year+'_'+quar]=df0

#本期淨利(淨損)

#%% 個別損益表（單季）

for year in range(2020,2021):
    for quar in range(3,4+1):
        year,quar=str(year),str(quar)
        df0=pd.DataFrame()
        for i in range(0,n,5):
            stock1=str(dict0['成分股']['code'][i])
            stock2=str(dict0['成分股']['code'][i+1])
            stock3=str(dict0['成分股']['code'][i+2])
            stock4=str(dict0['成分股']['code'][i+3])
            stock5=str(dict0['成分股']['code'][i+4])
            
            df=crawl_income_statement(stock1,stock2,stock3,stock4,stock5,year,quar)
            df.columns=[str(df.columns[i][0])+'_'+str(df.columns[i][1]) for i in range(len(df.columns))]
            #df.drop(['-_金額','-_％'],axis=1,inplace=True)
            df.set_index(df.columns[0],inplace=True)
            df=df.loc['稅後淨利',:]
            df=pd.DataFrame(df).T
            df0=pd.concat([df0,df],axis=1)
            
            time.sleep(30)
        dict0['個別損益表（單季）'+year+'_'+quar]=df0

#稅後淨利

#%% yahoo finance
import yfinance as yf

start='2009-12-01'
end='2021-04-30'
ticker=[str(i)+'.TW' for i in dict0['成分股']['code']]
df_yf=yf.download(ticker,start=start,end=end,auto_adjust=True,interval='1d')
n_close=len(df_yf['Close'].columns)
df_yf.columns=[str(df_yf.columns[i][0])+'_'+str(df_yf.columns[i][1]) for i in range(len(df_yf.columns))]

df_yf=df_yf.diff(axis=0,periods=59)/df_yf
df_yf=df_yf.iloc[:,0:n_close]
dict0['Return']=df_yf

#%%
import os
#os.makedirs('/Users/JasminePai/python/FINANCIAL_BIG_DATA/table')
os.chdir('/Users/JasminePai/python/FINANCIAL_BIG_DATA/table')
for i in dict0.keys():
    dict0[str(i)].to_csv(str(i)+'.csv')
