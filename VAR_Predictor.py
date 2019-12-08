#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:26:16 2019

@author: flo
"""

#%%

import urllib.request as url
import numpy as np
import pandas as pd


startdate = '2019-02-11'
enddate = '2019-02-15'

daterange = pd.date_range(startdate,enddate)
daterange = pd.DataFrame(data=daterange)
daterange.rename(columns={0: 'Date'}, inplace=True) 


#%%

#factor1

factor1url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%5eGSPC&outputsize=full&apikey=CAJH46WI0QYW2RGK&datatype=csv'
factor2url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NDAQ&outputsize=full&apikey=CAJH46WI0QYW2RGK&datatype=csv'
factor3url = 'https://www.quandl.com/api/v3/datasets/OPEC/ORB/data.csv?start_date='+startdate+'&end_date='+enddate+'&api_key=LwmAyD-1JHAzhMKKSvDX'
factor4url = 'https://www.quandl.com/api/v3/datasets/USTREASURY/YIELD.csv?start_date='+startdate+'&end_date='+enddate+'&api_key=LwmAyD-1JHAzhMKKSvDX'



alphaurls = {"Factor 1":factor1url,"Factor 2":factor2url}
quandlurls = {"Factor 3":factor3url,"Factor 4":factor4url}

#%%

#Creating DF for Factor 1 & 2

lst = []
for i in alphaurls:
    remoteFile = url.urlopen(alphaurls[i])
    html = remoteFile.read().decode('ascii').splitlines()
    b = pd.DataFrame(data=html)
    b = b[0].str.split(",", expand = True)
    b.columns = b.iloc[0]
    b = b[1:]
    final = b[['timestamp','close']]
    final = final[final['timestamp'] >= startdate]
    final = final[final['timestamp'] <= enddate]
    final.rename(columns={'close': i}, inplace=True) 
    final.rename(columns={'timestamp': 'Date'}, inplace=True) 
    final['Date'] = pd.to_datetime(final.Date)
    lst.append(final)
factor1_2 = pd.merge(lst[0], lst[1], how='outer', on=['Date'])    

#%%

merged = pd.merge(daterange, factor1_2, how='left', on=['Date'])
merged['Date'] = merged['Date'].dt.date

#%% Adding Factor 3

remoteFile = url.urlopen(factor3url)
html = remoteFile.read().decode('ascii').splitlines()
b = pd.DataFrame(data=html)
b = b[0].str.split(",", expand = True)
b.columns = b.iloc[0]
b = b[1:]
factor3 = b[b['Date'] >= startdate]
factor3 = b[b['Date'] <= enddate]
factor3.rename(columns={'Value': 'Factor 3'}, inplace=True) 
factor3 = factor3.sort_values(by='Date').reset_index()
factor3 = factor3.drop("index",1)
factor3['Date'] = pd.to_datetime(factor3.Date)
factor3['Date'] = factor3['Date'].dt.date

merged_1_2_3 = pd.merge(merged, factor3, how='left', on=['Date'])


#%%

#Factor 4 - taking only the bonds for one month

remoteFile = url.urlopen(factor4url)
html = remoteFile.read().decode('ascii').splitlines()
b = pd.DataFrame(data=html)
b = b[0].str.split(",", expand = True)
b.columns = b.iloc[0]
b = b[1:]
factor4 = b[b['Date'] >= startdate]
factor4 = b[b['Date'] <= enddate]
factor4 = factor4[['Date','1 MO']]
factor4.rename(columns={'1 MO': 'Factor 4'}, inplace=True) 
factor4['Date'] = pd.to_datetime(factor4.Date)
factor4['Date'] = factor4['Date'].dt.date

merged_1_2_3_4 = pd.merge(merged_1_2_3, factor4, how='left', on=['Date'])


#%%

#Replacing Weekends and public holidays with the previous values

P_merged_1_2_3_4 = merged_1_2_3_4.fillna(method='ffill')





#%%


symbols = ['PIH''FLWS','FCTY','FCCY','SRCE','FUBC','VNET','TWOU','DGLD','JOBS','EGHT','AVHI','SHLM','AAON','ASTM','ABAX','XLRN','ACTA','BIRT','MULT','YPRO','AEGR','MDRX','EPAX','DOX','UHAL','MTGE','CRMT','FOLD','BCOM','BOSC','HAWK','CFFI','CHRW','KOOL','HOTR','PLCE','JRJC','CHOP','HGSH','HTHT','IMOS','DAEG','DJCO','SATS','WATT','INBK','FTLB','QABA','GOOG']

apikey = 'O1TIWSKVY5BPDGS8'
symbol = 'AAON'

remoteFile = url.urlopen('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={0}&outputsize=full&apikey={1}&datatype=csv'.format(symbol, apikey))
html = remoteFile.read().decode('ascii').splitlines()

df = pd.DataFrame(data=html)
df = df[0].str.split(',', expand=True)
df.columns = df.iloc[0]
df = df[1:]

df_close = df[['timestamp','close']]

df_close = df_close[df_close['timestamp'] >= startdate]
df_close.rename(columns={'timestamp': 'Date'}, inplace=True) 


df_close = df_close.sort_values(by='Date').reset_index()
df_close = df_close.drop("index",1)


#%%
#Adding a custom daterange to account for holidays etc that are different in datasets

daterange = pd.date_range(startdate,enddate)
daterange = pd.DataFrame(data=daterange)
daterange.rename(columns={0: 'Date'}, inplace=True) 

df_close['Date'] = pd.to_datetime(df_close.Date)

instrument = pd.merge(daterange, df_close, how='left', on=['Date'])

instrument['Date'] = instrument['Date'].dt.date

instrument = instrument.fillna(method='ffill')




#%%

from sklearn.preprocessing import StandardScaler


#Predictor:


scaler = StandardScaler() 
model = "from MODEL"



def predictor(dataframe, instrument):
    X = dataframe['close']
    y = dataframe[['Factor 1','Factor 2','Factor 3','Factor 4']]
    X = scaler.fit_transform(X)
    model.fit(X, y)
    X_predict = instrument['close']
    X_predict = scaler.transform(X_predict)
    y_predict = model.predict(X_predict)
    return y_predict
    
    
