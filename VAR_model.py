#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 12:15:58 2019

@author: flo
"""

#%%

from VAR_Factors import merged_1_2_3_4
from VAR_Instruments import instrument
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

from sklearn import svm





#%%


dataset =  pd.merge(merged_1_2_3_4, instrument, how='inner', on=['Date'])

y = dataset['close']
X = dataset[['Factor 1','Factor 2','Factor 3','Factor 4']]

model = linear_model.Ridge()
#model3 = svm.SVR(kernel='linear', C=1000)
#model2 = RandomForestRegressor(n_estimators=500, n_jobs=-1)
#scaler = StandardScaler() 
cvStrategy = KFold(n_splits=6, shuffle=True, random_state=0)
scoring = 'r2' 
#X_Scaled = scaler.fit_transform(X)


scores = cross_val_score(model, X, y.values.flatten(), cv=cvStrategy, scoring=scoring)#
print(scores)
print(np.average(scores))

#%%





