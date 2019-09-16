# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 03:38:47 2019

@author: Иван
"""

import pandas as pd
import numpy as np

df_train=pd.read_csv('train1.csv',parse_dates=['timestamp'])
df_test=pd.read_csv('test1.csv',parse_dates=['timestamp'])
k=df_train['price_doc']
id_tr=df_train['id'].values
id_ts=df_test['id'].values
df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)
'Пометим данные'
df_test['istest']=1
df_train['istest']=0
data=pd.concat([df_train, df_test])

data1=data[data.columns[:60]]

print(data1.product_type.value_counts())
data1.product_type.fillna('Investment', inplace=True)
data1.product_type.replace({'Investment':0,'OwnerOccupier':1}, inplace=True)

data1.drop(['sub_area'], axis=1, inplace=True)
data1.thermal_power_plant_raion.replace({'no':0,'yes':1}, inplace=True)
data1.culture_objects_top_25.replace({'no':0,'yes':1}, inplace=True)
data1.incineration_raion.replace({'no':0,'yes':1}, inplace=True)
data1.oil_chemistry_raion.replace({'no':0,'yes':1}, inplace=True)
data1.radiation_raion.replace({'no':0,'yes':1}, inplace=True)
data1.railroad_terminal_raion.replace({'no':0,'yes':1}, inplace=True)
data1.big_market_raion.replace({'no':0,'yes':1}, inplace=True)
data1.nuclear_reactor_raion.replace({'no':0,'yes':1}, inplace=True)
data1.detention_facility_raion.replace({'no':0,'yes':1}, inplace=True)

n=1
while n<50:
    data1[data1.columns[n]].fillna(data1[data1.columns[n]].mean(), inplace=True)
    n+=1
    
#data1.info()
'следующий блок'

data2=data[data.columns[60:120]]

data2.water_1line.replace({'no':0,'yes':1}, inplace=True)
data2.big_road1_1line.replace({'no':0,'yes':1}, inplace=True)
data2.railroad_1line.replace({'no':0,'yes':1}, inplace=True)

n=0
while n<60:
    data2[data2.columns[n]].fillna(data2[data2.columns[n]].mean(), inplace=True)
    n+=1
#data2.info()

'следующий блок'
data3=data[data.columns[120:180]]
data3.ecology.replace({'poor':0,'satisfactory':1,'no data':1,
                       'good':2,'excellent':3}, inplace=True)
n=0
while n<60:
    data3[data3.columns[n]].fillna(data3[data3.columns[n]].mean(), inplace=True)
    n+=1
#data3.info()

'следующий блок'

data4=data[data.columns[180:240]]

n=0
while n<60:
    data4[data4.columns[n]].fillna(data4[data4.columns[n]].mean(), inplace=True)
    n+=1
#data4.info()
'следующий блок'

data5=data[data.columns[240:]]
n=0
while n<51:
    data5[data5.columns[n]].fillna(data5[data5.columns[n]].mean(), inplace=True)
    n+=1
data5.info()

for column in data2.columns:
    data1[column]=data2[column]
for column in data3.columns:
    data1[column]=data3[column]
for column in data4.columns:
    data1[column]=data4[column]
for column in data5.columns:
    data1[column]=data5[column]

train=data1[data1['istest']==0]
test=data1[data1['istest']==1]
test.drop(['istest'], axis=1, inplace=True)
train.drop(['istest'], axis=1, inplace=True)
train['price_doc']=k
train['id']=id_tr
test['id']=id_ts

train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)

