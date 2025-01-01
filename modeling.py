#財精四C 07155328白祐瑄
import os
import pandas as pd
#匯入資料
path='/Users/JasminePai/python/QUANTITATIVE FINANCE/期末報告/table/'

l_cash_flow=os.listdir(path+'現金流量表')
dict_cash_flow=dict()
for i in l_cash_flow:
    df=pd.read_csv(path+'現金流量表/'+i,index_col=[0])
    df.columns=[j.split('(')[1].split(')')[0] for j in  df.columns]
    df.index=['cash_flow']
    dict_cash_flow[i.split('）')[1].split('.')[0]]=df

l_income_s=os.listdir(path+'個別損益表')
dict_income_s=dict()
for i in l_income_s:
    df=pd.read_csv(path+'個別損益表/'+i,index_col=[0])
    df=df.iloc[:,[j for j in range(1,100,2)]]
    df.columns=[j.split('(')[1].split(')')[0] for j in  df.columns]
    df.index=['income_s']
    dict_income_s[i.split('）')[1].split('.')[0]]=df
    
l_roe=os.listdir(path+'ROE')
dict_roe=dict()
for i in l_roe:
    df=pd.read_csv(path+'ROE/'+i,index_col=[0])
    df.drop('代號.1',axis=1,inplace=True)
    df=df.T
    df.index=[j.split('Q')[0]+'_'+j.split('Q')[1] for j in df.index]
    df.columns=[str(j) for j in df.columns]
    for k in df.index:
        df0=pd.DataFrame(df.loc[k,:]).T
        df0.index=['roe']
        dict_roe['20'+k]=df0
        
l_ratio_table=os.listdir(path+'個別財務比率表')
dict_ratio_table=dict()
dict_ratio_table_index=dict()
for i in l_ratio_table:
    df=pd.read_csv(path+'個別財務比率表/'+i,index_col=[0])
    df.columns=[j.split('(')[1].split(')')[0] for j in df.columns]
    df_c=df.copy()
    df.index=[j for j in range(len(df.index))]
    dict_ratio_table_index[i.split('）')[1].split('.')[0]]={x:y for x,y in zip(df.index,df_c.index)}
    dict_ratio_table[i.split('）')[1].split('.')[0]]=df

#將技術指標資料取每季最後一日做為當季資料
def split_d_to_q(df_split,dict_split_q,index_name):
    for i in range(2010,2020+1):
        for j in range(1,4+1):
            if j==1:
                split_date=str(i)+'-'+'03'
                df=pd.DataFrame(df_split[split_date].iloc[-1,:]).T
                df.index=[index_name]
                dict_split_q[str(i)+'_'+str(j)]=df
            elif j==2:
                split_date=str(i)+'-'+'06'
                df=pd.DataFrame(df_split[split_date].iloc[-1,:]).T
                df.index=[index_name]
                dict_split_q[str(i)+'_'+str(j)]=df
            elif j==3:
                split_date=str(i)+'-'+'09'
                df=pd.DataFrame(df_split[split_date].iloc[-1,:]).T
                df.index=[index_name]
                dict_split_q[str(i)+'_'+str(j)]=df
            else:
                split_date=str(i)+'-'+'12'
                df=pd.DataFrame(df_split[split_date].iloc[-1,:]).T
                df.index=[index_name]
                dict_split_q[str(i)+'_'+str(j)]=df
 
#macd
df_macd=pd.read_csv(path+'macd.csv',index_col=[0],parse_dates=True)
df_macd.columns=[i.split('.')[0] for i in df_macd.columns]
dict_macd=dict()
split_d_to_q(df_macd,dict_macd,'macd')
#macdsignal
df_macdsignal=pd.read_csv(path+'macdsignal.csv',index_col=[0],parse_dates=True)
df_macdsignal.columns=[i.split('.')[0] for i in df_macdsignal.columns]
dict_macdsignal=dict()
split_d_to_q(df_macdsignal,dict_macdsignal,'macdsignal')
#macdhist
df_macdhist=pd.read_csv(path+'macdhist.csv',index_col=[0],parse_dates=True)
df_macdhist.columns=[i.split('.')[0] for i in df_macdhist.columns]
dict_macdhist=dict()
split_d_to_q(df_macdhist,dict_macdhist,'macdhist')
#sma
df_sma=pd.read_csv(path+'sma.csv',index_col=[0],parse_dates=True)
df_sma.columns=[i.split('.')[0] for i in df_sma.columns]
dict_sma=dict()
split_d_to_q(df_sma,dict_sma,'sma')
#ema
df_ema=pd.read_csv(path+'ema.csv',index_col=[0],parse_dates=True)
df_ema.columns=[i.split('.')[0] for i in df_ema.columns]
dict_ema=dict()
split_d_to_q(df_ema,dict_ema,'ema')
#wma
df_wma=pd.read_csv(path+'wma.csv',index_col=[0],parse_dates=True)
df_wma.columns=[i.split('.')[0] for i in df_wma.columns]
dict_wma=dict()
split_d_to_q(df_wma,dict_wma,'wma')
#rsi
df_rsi=pd.read_csv(path+'rsi.csv',index_col=[0],parse_dates=True)
df_rsi.columns=[i.split('.')[0] for i in df_rsi.columns]
dict_rsi=dict()
split_d_to_q(df_rsi,dict_rsi,'rsi')
#mom
df_mom=pd.read_csv(path+'mom.csv',index_col=[0],parse_dates=True)
df_mom.columns=[i.split('.')[0] for i in df_mom.columns]
dict_mom=dict()
split_d_to_q(df_mom,dict_mom,'mom')
#beta
df_beta=pd.read_csv(path+'beta.csv',index_col=[0],parse_dates=True)
df_beta.columns=[i.split('.')[0] for i in df_beta.columns]
dict_beta=dict()
split_d_to_q(df_beta,dict_beta,'beta')
#kd_d
df_kd_d=pd.read_csv(path+'kd_d.csv',index_col=[0],parse_dates=True)
df_kd_d.columns=[i.split('.')[0] for i in df_kd_d.columns]
dict_kd_d=dict()
split_d_to_q(df_kd_d,dict_kd_d,'kd_d')
#kd_k
df_kd_k=pd.read_csv(path+'kd_k.csv',index_col=[0],parse_dates=True)
df_kd_k.columns=[i.split('.')[0] for i in df_kd_k.columns]
dict_kd_k=dict()
split_d_to_q(df_kd_k,dict_kd_k,'kd_k')
#bb_upper
df_bb_upper=pd.read_csv(path+'bb_upper.csv',index_col=[0],parse_dates=True)
df_bb_upper.columns=[i.split('.')[0] for i in df_bb_upper.columns]
dict_bb_upper=dict()
split_d_to_q(df_bb_upper,dict_bb_upper,'bb_upper')
#bb_lower
df_bb_lower=pd.read_csv(path+'bb_lower.csv',index_col=[0],parse_dates=True)
df_bb_lower.columns=[i.split('.')[0] for i in df_bb_lower.columns]
dict_bb_lower=dict()
split_d_to_q(df_bb_lower,dict_bb_lower,'bb_lower')
#bb_middle
df_bb_middle=pd.read_csv(path+'bb_middle.csv',index_col=[0],parse_dates=True)
df_bb_middle.columns=[i.split('.')[0] for i in df_bb_middle.columns]
dict_bb_middle=dict()
split_d_to_q(df_bb_middle,dict_bb_middle,'bb_middle')
#adx
df_adx=pd.read_csv(path+'adx.csv',index_col=[0],parse_dates=True)
df_adx.columns=[i.split('.')[0] for i in df_adx.columns]
dict_adx=dict()
split_d_to_q(df_adx,dict_adx,'adx')

#return
df_return=pd.read_csv(path+'Return.csv',index_col=[0],parse_dates=True)
df_return.columns=[i.split('_')[1].split('.')[0] for i in df_return.columns]   
dict_return_train=dict()

'''
將報酬取對應下一季報酬，如：2010第1季特徵對應輸出變量為2010第2季報酬，
也就是2010-04-01到2010-06-30報酬。
'''
def split_d_to_q_return(df_split,dict_split_q,index_name):
    for i in range(2010,2020+1):
        for j in range(1,4+1):
            if j==1:
                split_date=str(i)+'-'+'06'
                df=pd.DataFrame(df_split[split_date].iloc[-1,:]).T
                df.index=[index_name]
                dict_split_q[str(i)+'_'+str(j)]=df
            elif j==2:
                split_date=str(i)+'-'+'09'
                df=pd.DataFrame(df_split[split_date].iloc[-1,:]).T
                df.index=[index_name]
                dict_split_q[str(i)+'_'+str(j)]=df
            elif j==3:
                split_date=str(i)+'-'+'12'
                df=pd.DataFrame(df_split[split_date].iloc[-1,:]).T
                df.index=[index_name]
                dict_split_q[str(i)+'_'+str(j)]=df
            elif j==4:
                split_date=str(i+1)+'-'+'03'
                df=pd.DataFrame(df_split[split_date].iloc[-1,:]).T
                df.index=[index_name]
                dict_split_q[str(i)+'_'+str(j)]=df

split_d_to_q_return(df_return,dict_return_train,'return')

#%% high to low 

dict_all_ori={'cash_flow':dict_cash_flow,
              'income_s':dict_income_s,
              'wma':dict_wma,'ema':dict_ema,
              'macd':dict_macd,'macdhist':dict_macdhist,'macdsignal':dict_macdsignal,
              'mom':dict_mom,
              'ratio_table':dict_ratio_table,
              'roe':dict_roe,'rsi':dict_rsi,'adx':dict_adx,
              'kd_d':dict_kd_d,'kd_k':dict_kd_k,
              'sma':dict_sma,'beta':dict_beta,
              'bb_lower':dict_bb_lower,'bb_middle':dict_bb_middle,'bb_upper':dict_bb_upper,
              'return':dict_return_train}

#將每季所有因子及報酬整理
l0=list()
def concat_q(year,quan,dictF):
    df=dict_all_ori[k][str(year)+'_'+str(quan)]
    l0.append(df)

dict_b_hl=dict()
for i in range(2010,2020+1):
    for j in range(1,4+1):
        l0=list()
        for k in dict_all_ori.keys():
            concat_q(i,j,k)
        dict_b_hl[str(i)+'_'+str(j)]=pd.concat(l0)

for i in dict_b_hl.keys():
    dict_b_hl[i]=dict_b_hl[i].T
    for j in dict_b_hl[i].columns:
        dict_b_hl[i][j]=pd.to_numeric(dict_b_hl[i][j],errors='coerce')

#%%
'''
將每季依據每個因子的計算值來對整個股票樣本做排序，
用以後續把由大到小的數值分成五組股票部位來進行等權重投資。
'''

dict_hl_code=dict()
def sort_values_by(df_sort):
    df0=df_sort.copy()
    df0.index=[j for j in range(1,len(df0.index)+1)]
    for i in df_sort.columns:
        df0[i]=df_sort.sort_values(by=[i],ascending=False).index
    return df0

for k in dict_b_hl.keys():
    dict_hl_code[k]=sort_values_by(dict_b_hl[k])

#%%
import copy
dict_return_hl=copy.deepcopy(dict_hl_code)
'''
把上一步原本是呈現每個因子的由大到小的股票代碼排序，
改成個股當季所對應的下一季季報酬。
'''

def sort_values_return(df_sort):
    df0=df_sort.copy()
    df0.index=[j for j in range(1,len(df0.index)+1)]
    for i in df_sort.columns:
        df0[i]=list(df_sort.sort_values(by=[i],ascending=False)['return'])
    return df0

for k in dict_return_hl.keys():
    dict_return_hl[k]=sort_values_return(dict_b_hl[k])
    
#%% return of portfolio

dict_return_p=dict()
portfolio_c=5
portfolio_member=int(len(dict_return_hl['2010_1'].index)/portfolio_c)

'''
計算由每個因子由高到底排序後，每季分成五組股票部位來進行等權重投資。
因為選定為0050成份股，所以應為每個因子下前1～10股一組，11～20一組，以此類推，做等權重投資。
'''

def portfolio_return(df_r_hl,df_r_p):
    for i in df_r_hl.columns:
        l_p_m_r=list()
        for j in range(1,len(df_r_hl[i])+1,portfolio_member):
            p_m_r=df_r_hl.loc[j:j+portfolio_member-1,i].mean()
            l_p_m_r.append(p_m_r)
        df_r_p[i]=l_p_m_r    
    return df_r_p

for k in dict_return_hl.keys():
    df_return_p=pd.DataFrame()
    dict_return_p[k]=portfolio_return(dict_return_hl[k],df_return_p)

#%% p_value
import scipy.stats as stats
'''
找出每季每個因子的投資組合的最高組及最低組進行t檢定，因此應該每季每個因子會獲得43個p-value。
'''

df_ttest_pvalue=pd.DataFrame('nan',index=[i for i in dict_return_p.keys()],columns=[j for j in dict_return_p['2010_1'].columns])
def t_test_pvalue(df_return_p,df_return_hl,df_ttest_value):
    for i in df_return_p.columns:
        ttest_1=df_return_p[i].idxmax()
        ttest_2=df_return_p[i].idxmin()
        ttest_1_array=df_return_hl.loc[ttest_1*portfolio_member+1:ttest_1*portfolio_member+portfolio_member,i]
        ttest_2_array=df_return_hl.loc[ttest_2*portfolio_member+1:ttest_2*portfolio_member+portfolio_member,i]
        p_value=stats.ttest_ind(a=ttest_1_array,b=ttest_2_array,equal_var=True).pvalue
        df_ttest_value[i][k]=p_value

l_pvalue=[i for i in dict_return_p.keys() if i !='2020_4']
for k in l_pvalue:
    t_test_pvalue(dict_return_p[k],dict_return_hl[k],df_ttest_pvalue)

'''
計算每個因子在43季中總共有多少p值小於0.1、大於0.1、及空缺值。
空缺值的原因為0050成份股的變動及財報資料的缺少。
'''
p0=0.1
df_ttest_pvalue_c=pd.DataFrame(0,index=['<=p0','>p0','nan'],columns=[i for i in df_ttest_pvalue.columns])
for i in df_ttest_pvalue.columns:
    df_ttest_pvalue_c[i]['<=p0']=sum(df_ttest_pvalue[i].astype(float)<=p0)
    df_ttest_pvalue_c[i]['>p0']=sum(df_ttest_pvalue[i].astype(float)>p0)
    df_ttest_pvalue_c[i]['nan']=len(l_pvalue)-sum(df_ttest_pvalue[i].astype(float)<=p0)-sum(df_ttest_pvalue[i].astype(float)>p0)

df_ttest_pvalue_c.drop(['return'],axis=1,inplace=True)

#%% features
'''
留下p值數量小於0.1前10名做為輸入模型的特徵。
並除去有缺失值的樣本。
'''
df_ttest_pvalue_c=df_ttest_pvalue_c.T
df_ttest_pvalue_c.sort_values(by=['<=p0'],inplace=True,ascending=False)
l_train_feature=[str(i) for i in df_ttest_pvalue_c['<=p0'].index[0:10]]

l_train_feature.append('return')

dict_train=dict()
for i in dict_b_hl.keys():
    dict_b_hl[i].columns=[str(i) for i in dict_b_hl[i].columns]
    dict_train[i]=dict_b_hl[i].loc[:,l_train_feature]
    dict_train[i]=dict_train[i].dropna(axis=0)

for i in dict_train.keys():
    dict_train[i].index=[str(j)+'_'+str(i) for j in dict_train[i].index]

#%% renew quan

dict_train_c=copy.deepcopy(dict_train)

'''
因原始文章中為對每季均進行預測，因此需要把預測季前的所有資料合併使用。
如預測2015第二季，訓練資料則由2010至2015第一季。
因此次報告只預測2020第四季，所以只會用到合併資料的最後一筆，
也就是2020第三季(其中為2010第一季至2020第3季資料)。
'''
dict_X_train=dict()
l_X_train=[i for i in dict_train_c.keys()]
l_X_train=l_X_train[0:-1]

for i in l_X_train:
    df0=pd.DataFrame()
    a=l_X_train.index(i)
    l_X_train0=l_X_train[0:a+1]
    dict_X_train[i]=pd.concat([dict_train_c[j] for j in l_X_train0])
    
#%%
#MLP
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler

minmax=MinMaxScaler()
activation='relu'
alpha=0.001 
solver='adam'
learning_rate_init=0.001 
max_iter=1000
tol=1e-6
early_stopping=True
MLP=MLPRegressor(hidden_layer_sizes=(64,64,64,64,64),
                 activation=activation,
                 alpha=alpha,
                 solver=solver,
                 learning_rate_init=learning_rate_init,
                 max_iter=max_iter,
                 tol=tol,
                 early_stopping=early_stopping
                 )

model=Pipeline([('scaler',minmax),
                ('regressor',MLP)
                ])
    
#%%
df_true_pred=pd.DataFrame()
df_true_pred['return_true']=dict_train_c['2020_4']['return']

final_pred_n=5
'''
進行5次模型訓練及預測，將5次結果加總平均為最終預測值。
'''

for i in range(final_pred_n):
    model.fit(dict_X_train['2020_3'].iloc[:,0:-1],dict_X_train['2020_3'].iloc[:,-1])
    df_true_pred['pred_'+str(i+1)]=model.predict(dict_train_c['2020_4'].iloc[:,0:-1])

df_true_pred_train=pd.DataFrame()
df_true_pred_train['return_true']=dict_X_train['2020_3']['return']
df_true_pred_train['pred']=model.predict(dict_X_train['2020_3'].iloc[:,0:-1])


l_final_pred=[]
for i in range(len(df_true_pred.index)):
    final_pred=df_true_pred.iloc[i,final_pred_n*(-1):].mean()
    l_final_pred.append(final_pred)
    
df_true_pred['final_pred']=l_final_pred

mse=metrics.mean_squared_error(df_true_pred['return_true'],df_true_pred['final_pred'])**0.5
print('MSE:'+str(mse))

#%%
'''
真實值與預測值的畫圖
'''
import matplotlib.pyplot as plt
fontsize = 14
plt.rcParams["figure.figsize"] = (6.4, 4.8)
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["axes.titlesize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize
plt.rcParams["legend.fontsize"] = fontsize

def plot_target_pred(y, y_pred, target, title):

    plt.figure()
    target_pred = target+"_pred"
    plt.plot(y, "b-", label=target)
    plt.plot(y_pred, "r-.", label=target_pred)
    plt.legend()
    plt.title(title)
    
    plt.figure()
    target_pred = target+"_pred"
    plt.plot(y, y_pred, ".")
    plt.xlabel(target)
    plt.ylabel(target_pred)
    plt.title(title)
    plt.show()
    
plot_target_pred(df_true_pred['return_true'],df_true_pred['final_pred'],'return','true v.s pred')

plot_target_pred(df_true_pred_train['return_true'],df_true_pred_train['pred'],'return','true v.s pred')


