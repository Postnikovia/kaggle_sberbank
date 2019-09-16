# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 00:32:39 2019

@author: Иван
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
import matplotlib.pyplot as plt 
"Импорт данных"
df_train=pd.read_csv('train.csv',parse_dates=['timestamp'])
df_test=pd.read_csv('test.csv',parse_dates=['timestamp'])
df_macro=pd.read_csv('macro2.csv',parse_dates=['timestamp'])
df_macro.drop([df_macro.columns[0]], axis=1, inplace=True)
ylog_train_all=np.log1p(df_train['price_doc'].values)
"Целевая переменная"
k=df_train['price_doc'].values
ax = df_train['price_doc'].hist(bins=140)
"Сохраняем id и очищаем от не информативных столбцов выборки"
id_test = df_test['id']
df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)
'Помечаем выборки'
num_train = len(df_train)
df_train['istest']=0
df_test['istest']=1
"Склеиваем выборки между собой и добавляем экономические показатели"
df_all = pd.concat([df_train, df_test])
df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')
print(df_all.shape)
"Добавляем номер месяца "
month_year = df_all.timestamp.dt.month
df_all['month_year_cnt'] = month_year
df_all['dow'] = df_all.timestamp.dt.dayofweek

"Добавляем новые фичи"
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

df_all.drop(['timestamp'], axis=1, inplace=True)



df_values =df_all.copy()

df_values.info()
"Разделяем выборки"
train=df_values[df_values['istest']==0]
test=df_values[df_values['istest']==1]
train.drop(['istest'], axis=1, inplace=True)
test.drop(['istest'], axis=1, inplace=True)
"Оставались пустые значения и это помогло решить проблему"
train['price_doc']=k
train.dropna(inplace=True)

k=train['price_doc'].values

train.drop(['price_doc'],axis=1,  inplace=True)
""


xgb_params = {
    'eta': 0.03,
    'max_depth': 7,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

"Реализуем алгоритм"
dtrain = xgb.DMatrix(train,k)
dtest = xgb.DMatrix(test)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

output.to_csv('xgbSub.csv', index=False)


