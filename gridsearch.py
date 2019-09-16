# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 02:24:31 2019

@author: Иван
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt 

df_train=pd.read_csv('train.csv',parse_dates=['timestamp'])
df_test=pd.read_csv('test.csv',parse_dates=['timestamp'])
df_macro=pd.read_csv('macro2.csv',parse_dates=['timestamp'])
df_macro.drop([df_macro.columns[0]], axis=1, inplace=True)
ylog_train_all=np.log1p(df_train['price_doc'].values)
k=df_train['price_doc'].values
ax = df_train['price_doc'].hist(bins=140)

id_test = df_test['id']
df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

num_train = len(df_train)
df_train['istest']=0
df_test['istest']=1
df_all = pd.concat([df_train, df_test])
df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')
print(df_all.shape)

month_year = df_all.timestamp.dt.month
df_all['month_year_cnt'] = month_year
df_all['dow'] = df_all.timestamp.dt.dayofweek


df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

df_all.drop(['timestamp'], axis=1, inplace=True)

df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)

df_values.info()

train=df_values[df_values['istest']==0]
test=df_values[df_values['istest']==1]
train.drop(['istest'], axis=1, inplace=True)
test.drop(['istest'], axis=1, inplace=True)
train['price_doc']=k
train.dropna(inplace=True)
k=train['price_doc'].values
train.drop(['price_doc'],axis=1,  inplace=True)
xgb_params = {
    'max_depth': [6,8,9,10,11],
    'min_child_weight':[4,5,6],
}

from sklearn.model_selection import GridSearchCV
xgb1=xgb.XGBRegressor()
parameters = {'nthread':[4],
              'objective':['reg:linear'],
              'learning_rate': [.03], 
              'max_depth': [7],
              'min_child_weight': [3,4,5],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [100,250,500]}
#0.03, 7
grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)
grid.fit(train, k)

print(grid.best_params_)
print(grid.best_score_)

