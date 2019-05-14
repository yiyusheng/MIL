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
data = pd.read_csv(home+'/Data/backblaze/data_bb',sep=',')

data_names = data.columns.values
data_names_meta = data_names[0:5]
data_names_attr = data_names[6:len(data_names)]

sta_null = data[data_names_attr].isnull().sum()/len(data)*100
valid_names = sta_null.index[sta_null<=10].values

data_filter = data[data_names_meta.tolist()+valid_names.tolist()]
