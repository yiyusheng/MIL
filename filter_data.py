#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:17:42 2019

@author: yiyusheng

filter data
"""

import os
import pandas as pd

path = os.getenv("HOME")+'/Data/backblaze/'
data = pd.read_csv(path+'data_bb',sep=',')

data_names = data.columns.values
data_names_meta = data_names[0:5]
data_names_attr = data_names[6:len(data_names)]

# %% filter and save
sta_null = data[data_names_attr].isnull().sum()/len(data)*100
valid_names = sta_null.index[sta_null<=10].values

data_filter = data[data_names_meta.tolist()+valid_names.tolist()]
data_filter_info = data_filter.info()
data_filter_desc = data_filter.describe()

model_count = data_filter['model'].value_counts()/len(data_filter)
valid_model = model_count[0:6]
data_filter_major_models = data_filter[data_filter['model'].isin(valid_model.index.values)]
for m in valid_model.index.values:
    data_model = data_filter_major_models[data_filter_major_models['model']==m]
    data_model.to_csv(path+m,index=0)
    
# %% statistic by model 
