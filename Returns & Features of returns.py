#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:50:42 2019

@author: flo
"""


#%%

import urllib.request as url
import numpy as np
import pandas as pd


#%%

#f1, f2, f3, f4 = np.random.multivariate_normal(mean, cov)
#%%

#factor1

factor1url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%5eGSPC&outputsize=full&apikey=CAJH46WI0QYW2RGK&datatype=csv'
factor2url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=NDAQ&outputsize=full&apikey=CAJH46WI0QYW2RGK&datatype=csv'
factor3url = 'https://www.quandl.com/api/v3/datasets/OPEC/ORB/data.csv?start_date=2007-12-31&end_date=2019-01-31&api_key=LwmAyD-1JHAzhMKKSvDX'
factor4url = 'https://www.quandl.com/api/v3/datasets/USTREASURY/YIELD.csv?start_date=2007-12-31&end_date=2019-01-31&api_key=LwmAyD-1JHAzhMKKSvDX'



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
factor4 = factor4[['Date','1 MO']]
factor4.rename(columns={'1 MO': 'Factor 4'}, inplace=True) 
factor4['Date'] = pd.to_datetime(factor4.Date)
factor4['Date'] = factor4['Date'].dt.date

merged_1_2_3_4 = pd.merge(merged_1_2_3, factor4, how='left', on=['Date'])


#%%

#Replacing Weekends and public holidays with the previous values



merged_1_2_3_4 = merged_1_2_3_4.fillna(method='ffill')

X= merged_1_2_3_4

#%% Calcualting returns of each factor

X['Factor 1']= pd.to_numeric(X['Factor 1'])
X['Factor 2']= pd.to_numeric(X['Factor 2'])
X['Factor 3']= pd.to_numeric(X['Factor 3'])
X['Factor 4']= pd.to_numeric(X['Factor 4'])

#fact1= np.array(X['Factor 1'])
#fact2= np.array(X['Factor 2'])
#fact3= np.array(X['Factor 3'])
#fact4= np.array(X['Factor 3'])
#
#fact1= fact1[0::5]
#fact2= fact2[0::5]
#fact3= fact3[0::5]
#fact4= fact4[0::5]
#
#Retfact1 = []
#for i in range(1, len(fact1)):
#  returns = (fact1[i] - fact1[i-1])/fact1[i-1]
#  Retfact1.append(returns)
#  
#Retfact2 = []
#for i in range(1, len(fact2)):
#    
#        returns = (fact2[i] - fact2[i-1])/fact2[i-1]
#        Retfact2.append(returns)
#  
#Retfact3 = []
#for i in range(1, len(fact3)):
#        returns = (fact3[i] - fact3[i-1])/fact3[i-1]
#        Retfact3.append(returns)
#  
#Retfact4 = []
#for i in range(1, len(fact4)):
#        returns = (fact4[i] - fact4[i-1])/fact4[i-1]
#        Retfact4.append(returns)
#        
#Returnsfactor1 = pd.DataFrame(Retfact1)
#Returnsfactor2 = pd.DataFrame(Retfact2)
#Returnsfactor3 = pd.DataFrame(Retfact3)
#Returnsfactor4 = pd.DataFrame(Retfact4)
#
#X['Returnsfactor1'] = Returnsfactor1
#X['Returnsfactor2'] = Returnsfactor2
#X['Returnsfactor3'] = Returnsfactor3
#X['Returnsfactor4'] = Returnsfactor4
##%% Squaring factors returns
#
#X['FactorRe 1 S'] = Returnsfactor1.apply(lambda x: x**2)
#
#X['FactorRe 2 S'] = Returnsfactor2.apply(lambda x: x**2)
#
#X['FactorRe 3 S'] = Returnsfactor3.apply(lambda x: x**2)
#
#X['FactorRe 4 S'] = Returnsfactor4.apply(lambda x: x**2)
#
##%% Rooting factor returns
#
#X['FactorRe 1 R'] = Returnsfactor1.apply(lambda x: x** (1/2))
#
#X['FactorRe 2 R'] = Returnsfactor2.apply(lambda x: x** (1/2))
#
#X['FactorRe 3 R'] = Returnsfactor3.apply(lambda x: x** (1/2))
#
#X['FactorRe 4 R']= Returnsfactor4.apply(lambda x: x** (1/2))
#


##%% Adding featuralisation of the factors into one unique value
#
#TotalFact1 = X['Factor 1']+ X['Factor 1 S'] + X['Factor 1 R']
#TotalFact2 = X['Factor 2']+ X['Factor 2 S'] + X['Factor 2 R']
#TotalFact3 = X['Factor 3']+ X['Factor 3 S'] + X['Factor 3 R']
#TotalFact4 = X['Factor 4']+ X['Factor 4 S'] + X['Factor 4 R']
#
##%% Adding this unique value into  X dataframe
#
#X['SumFact1'] = TotalFact1
#X['SumFact2'] = TotalFact2
#X['SumFact3'] = TotalFact3
#X['SumFact4'] = TotalFact4  
#

#%% Calculating returns of each factor based on the sum of featuralised factors
#import numpy as np
#from numpy import mean
#from numpy import cov
#
#
#
#X['Weekday'] = pd.DatetimeIndex(X['Date']).weekday
#X['Weekday'] = X['Weekday'].map(lambda x: 0 if x == 5.0 or x == 6.0 else 1)
#
#X['Weekend'] = X['Weekday'].map(lambda x: 1 if x == 5.0 or x == 6.0 else 0)
#
#      
#Listfact1= np.array(X['SumFact1'])
#Listfact2= np.array(X['SumFact2'])
#Listfact3= np.array(X['SumFact3'])
#Listfact4= np.array(X['SumFact4'])
#
#Listfact1= Listfact1[0::5]
#Listfact2= Listfact2[0::5]
#Listfact3= Listfact3[0::5]
#Listfact4= Listfact4[0::5]
#
#
#Retfeat1 = []
#for i in range(1, len(Listfact1)):
#    returns = (Listfact1[i] - Listfact1[i-1])/Listfact1[i-1]
#    Retfact1.append(returns)
#  
#Retfeat2 = []
#for i in range(1, len(Listfact2)):
#        returns = (Listfact2[i] - Listfact2[i-1])/Listfact2[i-1]
#        Retfeat2.append(returns)
#  
#Retfeat3 = []
#for i in range(1, len(Listfact3)):
#  returns = (Listfact3[i] - Listfact3[i-1])/Listfact3[i-1]
#  Retfeat3.append(returns)
#  
#Retfeat4 = []
#for i in range(1, len(Listfact4)):
#  returns = (Listfact4[i] - Listfact4[i-1])/Listfact4[i-1]
#  Retfeat4.append(returns)
#


#%% Creating Dataframe for factor returns
#  
#Returnsdf1 = pd.DataFrame(Retfact1)
#Returnsdf2 = pd.DataFrame(Retfact2)
#Returnsdf3 = pd.DataFrame(Retfact3)
#Returnsdf4 = pd.DataFrame(Retfact4)
#
#Returnsdf= pd.concat([Returnsdf1,Returnsdf2,Returnsdf3,Returnsdf4],axis=1, sort=False)

#%% Calcultating returns of each factor

returns_factors = X.reset_index()
returns_factors = returns_factors[['Date', 'Factor 1', 'Factor 2', 'Factor 3', 'Factor 4']]



for index, row in returns_factors.iterrows():
        if index < 5:
            continue
        else:    
            returns_factors.loc[index, 'Return1'] = (returns_factors.loc[index, 'Factor 1'] - returns_factors.loc[index-5, 'Factor 1']) / returns_factors.loc[index-5, 'Factor 1']
for index, row in returns_factors.iterrows():
        if index < 5:
            continue
        else:    
            returns_factors.loc[index, 'Return2'] = (returns_factors.loc[index, 'Factor 2'] - returns_factors.loc[index-5, 'Factor 2']) / returns_factors.loc[index-5, 'Factor 2']

for index, row in returns_factors.iterrows():
        if index < 5:
            continue
        else:    
            returns_factors.loc[index, 'Return3'] = (returns_factors.loc[index, 'Factor 3'] - returns_factors.loc[index-5, 'Factor 3']) / returns_factors.loc[index-5, 'Factor 3']

for index, row in returns_factors.iterrows():
        if index < 5:
            continue
        else:    
            returns_factors.loc[index, 'Return4'] = (returns_factors.loc[index, 'Factor 4'] - returns_factors.loc[index-5, 'Factor 4']) / returns_factors.loc[index-5, 'Factor 4']


#%%
            
returns_factors = returns_factors.dropna().reset_index()
final_factors =   returns_factors[['Date','Return1','Return2','Return3','Return4']]

#%% Preparing returns dataframe for the distribution test
Returnsfactor1 = final_factors['Return1']
Returnsfactor2 = final_factors['Return2']
Returnsfactor3 = final_factors['Return3']
Returnsfactor4 = final_factors['Return4']


#%% Proving that the each factor returns are normally distributed via Shapiro test
#
from scipy import stats
import matplotlib.pyplot as plt

Shapirotest1= (stats.shapiro (Returnsfactor1))
print (Shapirotest1[0])
print ("this is W static")
print (Shapirotest1[1])
print ("this is p value")

Shapirotest2= (stats.shapiro (Returnsfactor2))
print (Shapirotest2[0])
print ("this is W static")
print (Shapirotest2[1])
print ("this is p value")

Shapirotest3= (stats.shapiro (Returnsfactor3))
print (Shapirotest3[0])
print ("this is W static")
print (Shapirotest3[1])
print ("this is p value")

Shapirotest4= (stats.shapiro (Returnsfactor4))
print (Shapirotest4[0])
print ("this is W static")
print (Shapirotest4[1])
print ("this is p value")

histFact1 = Returnsfactor1.hist(bins=50)
histFact2 = Returnsfactor2.hist(bins=50)
histFact3 = Returnsfactor3.hist(bins=50)
histFact4 = Returnsfactor4.hist(bins=50)


#%%Squaring factors returns

final_factors['Return1sq'] = final_factors['Return1'].apply(lambda x: x**2)
#
final_factors['Return2sq']= final_factors['Return2'].apply(lambda x: x**2)
#
final_factors['Return3sq']= final_factors['Return3'].apply(lambda x: x**2)
#
final_factors['Return4sq']= final_factors['Return4'].apply(lambda x: x**2)
#
#%% Rooting factor returns
#
final_factors['Return1rt'] = final_factors['Return1'].apply(lambda x: x** (1/2))
#
final_factors['Return2rt'] = final_factors['Return2'].apply(lambda x: x** (1/2))
#
final_factors['Return3rt'] = final_factors['Return3'].apply(lambda x: x** (1/2))
#
final_factors['Return4rt'] = final_factors['Return4'].apply(lambda x: x** (1/2))

#%% Adjusting the squares of the returns for the sign
#AdjustedReturns1, AdjustedReturns2,AdjustedReturns3, AdjustedReturns4 are the 
#final returns adjusted for the sign


signs1=[]

final_factors['Return1'].tolist()

for x in final_factors['Return1']:
    if x > 0:
        signs1.append(1)
    if x== 0:
        signs1.append(1)
    if x< 0:
        signs1.append (-1)
    
signs1 = np.array(signs1)
returns1 = np.array(final_factors['Return1sq'])


AdjustedReturns1 = signs1*returns1

signs2=[]

final_factors['Return2'].tolist()
for x in final_factors['Return2']:
    if x > 0:
        signs2.append(1)
    if x== 0:
        signs2.append(1)
    if x< 0:
        signs2.append (-1)


signs2 = np.array(signs2)
returns2 = np.array(final_factors['Return2sq'])


AdjustedReturns2 = signs2*returns2

signs3=[]

final_factors['Return3'].tolist()
for x in final_factors['Return3']:
    if x > 0:
        signs3.append(1)
    if x== 0:
        signs3.append(1)
    if x< 0:
        signs3.append (-1)


signs3 = np.array(signs3)
returns3= np.array(final_factors['Return3sq'])


AdjustedReturns3 = signs3*returns3

signs4=[]

final_factors['Return4'].tolist()
for x in final_factors['Return4']:
    if x > 0:
        signs4.append(1)
    if x== 0:
        signs4.append(1)
    if x< 0:
        signs4.append (-1)


signs4 = np.array(signs4)
returns4= np.array(final_factors['Return4sq'])


AdjustedReturns4 = signs4*returns4

#%%Adjusting roots of the returns for the sign

#ReturnsRT1, ReturnsRT2,ReturnsRT3, ReturnsRT4 are the 
#final returns adjusted for the sign

signs1=[]

final_factors['Return1'].tolist()

for x in final_factors['Return1']:
    if x > 0:
        signs1.append(1)
    if x== 0:
        signs1.append(1)
    if x< 0:
        signs1.append (-1)
    
signs1 = np.array(signs1)
RT1 = np.array(final_factors['Return1rt'])


ReturnsRT1 = signs1*RT1

signs2=[]

final_factors['Return2'].tolist()
for x in final_factors['Return2']:
    if x > 0:
        signs2.append(1)
    if x== 0:
        signs2.append(1)
    if x< 0:
        signs2.append (-1)


signs2 = np.array(signs2)
RT2 = np.array(final_factors['Return2rt'])


ReturnsRT2 = signs2*RT2

signs3=[]

final_factors['Return3'].tolist()
for x in final_factors['Return3']:
    if x > 0:
        signs3.append(1)
    if x== 0:
        signs3.append(1)
    if x< 0:
        signs3.append (-1)


signs3 = np.array(signs3)
RT3= np.array(final_factors['Return3rt'])


ReturnsRT3 = signs3*RT3

signs4=[]

final_factors['Return4'].tolist()
for x in final_factors['Return4']:
    if x > 0:
        signs4.append(1)
    if x== 0:
        signs4.append(1)
    if x< 0:
        signs4.append (-1)


signs4 = np.array(signs4)
RT4= np.array(final_factors['Return4rt'])


ReturnsRT4 = signs4*RT4

