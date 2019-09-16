# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 01:03:05 2019

@author: Иван
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

macro=pd.read_csv('macro.csv')


#Кол-во заполненых значений <1100 идут в утиль
time=macro['timestamp']
macro.drop(['modern_education_share','old_education_build_share','provision_retail_space_sqm',
            'provision_retail_space_modern_sqm','child_on_acc_pre_school'], axis=1, inplace=True)
macro.drop(['timestamp'], axis=1,inplace=True)
n_max=macro.shape[1]
n=0
while n<n_max:
    macro[macro.columns[n]].fillna(macro[macro.columns[n]].mean(), inplace=True)
    n+=1

from sklearn.decomposition import PCA
pca=PCA()
pca.fit(macro)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_max)]
feature=most_important[0:3]+most_important[4:11]
macro2=pd.DataFrame(time)
name_list=[]
for i in feature:
    name=macro[macro.columns[i]].name
    macro2[name]=macro[macro.columns[i]]
    name_list.append(name)

macro2.to_csv('macro2.csv')