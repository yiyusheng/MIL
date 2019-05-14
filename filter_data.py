#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:17:42 2019

@author: yiyusheng

filter data
"""

import pandas as pd
import os

home = os.getenv("HOME")
path_data = '/Data/backblaze/'
data = pd.read_csv(home+path_data+'data_2013',sep=',')

data_names = data.columns.values
data_names_meta = data_names[0:5]
data_names_attr = data_names[6:len(data_names)]

sta_null = data[data_names_attr].isnull().sum()/len(data)*100
valid_names = sta_null.index[sta_null<=10].values

data_filter = data[data_names_meta.tolist()+valid_names.tolist()]
data_filter.info()
data_filter.describe()

model_count = data_filter['model'].value_counts()/len(data_filter)
valid_model = model_count[0:3]
for m in valid_model.index.values:
    data_model = data_filter[data_filter['model']==m]
    data_model.to_csv(home+path_data+m,index=0)