#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:52:57 2019

@author: flo
"""

#%%
import pandas as pd
import urllib.request as url


symbols = ['PIH','FLWS','FCTY','FCCY','SRCE','FUBC','VNET','TWOU','DGLD','JOBS','EGHT','AVHI','SHLM','AAON','ABAX','XLRN','ACTA','BIRT','MULT','YPRO','AEGR','MDRX','EPAX','DOX','UHAL','MTGE','CRMT','FOLD','BCOM','BOSC','HAWK','CFFI','CHRW','KOOL','HOTR','PLCE','JRJC','CHOP','HGSH','HTHT','IMOS','DAEG','DJCO','SATS','WATT','INBK','FTLB','QABA','GOOG']
apikey = 'O1TIWSKVY5BPDGS8'
symbol = 'FCTY'

remoteFile = url.urlopen('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={0}&outputsize=full&apikey={1}&datatype=csv'.format(symbol, apikey))
html = remoteFile.read().decode('ascii').splitlines()

df = pd.DataFrame(data=html)
df = df[0].str.split(',', expand=True)
df.columns = df.iloc[0]
df = df[1:]

df_close = df[['timestamp','close']]

df_close = df_close[df_close['timestamp'] >= '2007-12-31']
df_close.rename(columns={'timestamp': 'Date'}, inplace=True) 


df_close = df_close.sort_values(by='Date').reset_index()
df_close = df_close.drop("index",1)


#%%
#Adding a custom daterange to account for holidays etc that are different in datasets



daterange = pd.date_range('2007-12-31','2019-01-31')
daterange = pd.DataFrame(data=daterange)
daterange.rename(columns={0: 'Date'}, inplace=True) 

df_close['Date'] = pd.to_datetime(df_close.Date)

instrument = pd.merge(daterange, df_close, how='left', on=['Date'])

instrument['Date'] = instrument['Date'].dt.date

instrument = instrument.fillna(method='ffill')


#%%

def calc_inst(inst):
    remoteFile = url.urlopen('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={0}&outputsize=full&apikey={1}&datatype=csv'.format(inst, apikey))
    html = remoteFile.read().decode('ascii').splitlines()
    df = pd.DataFrame(data=html)
    df = df[0].str.split(',', expand=True)
    df.columns = df.iloc[0]
    df = df[1:]
    df_close = df[['timestamp','close']]
    df_close = df_close[df_close['timestamp'] >= '2007-12-31']
    df_close.rename(columns={'timestamp': 'Date'}, inplace=True) 
    df_close = df_close.sort_values(by='Date').reset_index()
    df_close = df_close.drop("index",1)
    daterange = pd.date_range('2007-12-31','2019-01-31')
    daterange = pd.DataFrame(data=daterange)
    daterange.rename(columns={0: 'Date'}, inplace=True) 

    df_close['Date'] = pd.to_datetime(df_close.Date)

    instrument = pd.merge(daterange, df_close, how='left', on=['Date'])

    instrument['Date'] = instrument['Date'].dt.date

    instrument = instrument.fillna(method='ffill')
    # dropping NaNs afterwards so that we can start with the first available date
    instrument = instrument.dropna()
    return instrument

def calc_all():
    values = {}
    for inst in symbols:
        print(inst)
        df = calc_inst(inst)
        values[inst] = df   
    return values    
    






