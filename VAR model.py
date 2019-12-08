# -*- coding: utf-8 -*-


#%%

import seaborn as sns
import numpy as np
import pandas as pd

import urllib.request as url

#remoteFile = url.urlopen('https://www.quandl.com/api/v3/datasets/OPEC/ORB/data.csv?start_date=2008-01-01&end_date=2019-01-31&api_key=LwmAyD-1JHAzhMKKSvDX')
#html_OPEC = remoteFile.read().decode('ascii').splitlines()
#print(html_OPEC)
#dfOPEC = pd.DataFrame(data=html_OPEC)
#dfOPEC = dfOPEC[0].str.split(',', expand=True)
#dfOPEC.columns = dfOPEC.iloc[0]
#dfOPEC = dfOPEC[1:]
#dfOPEC
#
#
#
#remoteFile = url.urlopen('https://www.quandl.com/api/v3/datasets/USTREASURY/YIELD.csv?start_date=2008-01-01&end_date=2019-01-31&api_key=LwmAyD-1JHAzhMKKSvDX')
#html_US= remoteFile.read().decode('ascii').splitlines()
#print(html_US)
#dfUS = pd.DataFrame(data=html_US)
#dfUS = dfUS[0].str.split(',', expand=True)
#dfUS.columns = dfUS.iloc[0]
#dfUS = dfUS[1:]
#dfUS
#
#remoteFile = url.urlopen('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%5eGSPC&outputsize=full&apikey=CAJH46WI0QYW2RGK')
#html = remoteFile.read().decode('ascii').splitlines()
#print(html)
#dfhtml = pd.DataFrame(data=html)
#dfhtml= dfhtml [0].str.split(',', expand=True)
#dfhtml.columns = dfhtml.iloc[0]
#dfhtml= dfhtml[1:]
#dfhtml
#
#remoteFile = url.urlopen('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NDAQ&outputsize=full&apikey=CAJH46WI0QYW2RGK')
#html_NASDAQ = remoteFile.read().decode('ascii').splitlines()
#print(html_NASDAQ)
#dfNASDAQ = pd.DataFrame(html_NASDAQ)
#dfNASDAQ = dfNASDAQ[0].str.split(',', expand=True)
#dfNASDAQ.columns = dfNASDAQ.iloc[0]
#dfNASDAQ = dfNASDAQ[1:]
#dfNASDAQ

#%%

#f1, f2, f3, f4 = np.random.multivariate_normal(mean, cov)
#%%


factor1url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%5eGSPC&outputsize=full&apikey=CAJH46WI0QYW2RGK&datatype=csv'
factor2url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NDAQ&outputsize=full&apikey=CAJH46WI0QYW2RGK&datatype=csv'
factor3url = 'https://www.quandl.com/api/v3/datasets/OPEC/ORB/data.csv?start_date=2007-12-31&end_date=2019-01-31&api_key=LwmAyD-1JHAzhMKKSvDX'
factor4url = 'https://www.quandl.com/api/v3/datasets/USTREASURY/YIELD.csv?start_date=2007-12-31&end_date=2019-01-31&api_key=LwmAyD-1JHAzhMKKSvDX'


alphaurls = {"Factor 1":factor1url,"Factor 2":factor2url}
quandlurls = {"Factor 3":factor3url,"Factor 4":factor4url}

#%%
import math
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
    final = final[final['timestamp'] >= '2007-12-31']
    final.rename(columns={'close': i}, inplace=True) 
    final.rename(columns={'timestamp': 'Date'}, inplace=True) 
    final['Date'] = pd.to_datetime(final.Date)
    lst.append(final)
factor1_2 = pd.merge(lst[0], lst[1], how='outer', on=['Date'])    

#%%
#Adding a custom daterange to account for holidays etc that are different in datasets

daterange = pd.date_range('2007-12-31','2019-01-31')
daterange = pd.DataFrame(data=daterange)
daterange.rename(columns={0: 'Date'}, inplace=True) 


merged = pd.merge(daterange, factor1_2, how='left', on=['Date'])
merged['Date'] = merged['Date'].dt.date

#%% Adding Factor 3

remoteFile = url.urlopen(factor3url)
html = remoteFile.read().decode('ascii').splitlines()
b = pd.DataFrame(data=html)
b = b[0].str.split(",", expand = True)
b.columns = b.iloc[0]
b = b[1:]
factor3 = b[b['Date'] >= '2007-12-31']
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
factor4 = b[b['Date'] >= '2007-12-31']
factor4.rename(columns={'Value': 'Factor 4'}, inplace=True) 
factor4 = factor4[['Date','1 MO']]
factor4['Date'] = pd.to_datetime(factor4.Date)
factor4['Date'] = factor4['Date'].dt.date

merged_1_2_3_4 = pd.merge(merged_1_2_3, factor4, how='left', on=['Date'])

#%%

#Replacing Weekends and public holidays with the previous values

merged_1_2_3_4 = merged_1_2_3_4.fillna(method='ffill')


#%% Instruments

import pandas as pd
import urllib.request as url

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

df_close = df_close[df_close['timestamp'] >= '2007-12-31']
df_close.rename(columns={'timestamp': 'Date'}, inplace=True) 


df_close = df_close.sort_values(by='Date').reset_index()
df_close = df_close.drop("index",1)


df_close['Date'] = pd.to_datetime(df_close.Date)

instrument = pd.merge(daterange, df_close, how='left', on=['Date'])

instrument['Date'] = instrument['Date'].dt.date

instrument = instrument.fillna(method='ffill')

#%%
#Adding a custom daterange to account for holidays etc that are different in datasets

daterange = pd.date_range('2007-12-31','2019-01-31')
daterange = pd.DataFrame(data=daterange)
daterange.rename(columns={0: 'Date'}, inplace=True) 

df_close['Date'] = pd.to_datetime(df_close.Date)

instrument = pd.merge(daterange, df_close, how='left', on=['Date'])

instrument['Date'] = instrument['Date'].dt.date

instrument = instrument.fillna(method='ffill')




#%% Prepare a linear regression model for estimating future prices of the instruments based on the factors.
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


merged_1_2_3_4 = merged_1_2_3_4.drop("Date",1)
instrument = instrument.drop("Date",1)
X = merged_1_2_3_4
y = instrument
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#%% factorization of features, Squared

import numpy as np
import pandas as pd

X= merged_1_2_3_4
y = instrument

X['Factor 1 S'] = pd.to_numeric(X['Factor 1']).apply(lambda x: x**2)

X['Factor 2 S'] = pd.to_numeric(X['Factor 2']).apply(lambda x: x**2)

X['Factor 3 S'] = pd.to_numeric(X['Factor 3']).apply(lambda x: x**2)

X['1 MO S'] = pd.to_numeric(X['1 MO']).apply(lambda x: x**2)
#%% Root of the features added to the dataframe

X['Factor 1 R'] = pd.to_numeric(X['Factor 1']).apply(lambda x: x** (1/2))

X['Factor 2 R'] = pd.to_numeric(X['Factor 2']).apply(lambda x: x** (1/2))

X['Factor 3 R'] = pd.to_numeric(X['Factor 3']).apply(lambda x: x** (1/2))

X['1 MO R']= pd.to_numeric(X['1 MO']).apply(lambda x: x** (1/2))


#%%

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
#%%


