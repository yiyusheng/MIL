#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 09:34:27 2019

@author: yiyusheng

Normalize data
Select feature by z-test
k-fold group data
"""
import os,time
import numpy as np
import pandas as pd
import scipy
from merge_data import load_file,get_colnames 

def normalize_data(data):
    [name_meta,name_smart] = get_colnames(data)    
    data[name_smart]=data[name_smart].apply(lambda x:(x-x.min())/(x.max()-x.min()))
    return data

def add_tia(data):
    [name_meta,name_smart] = get_colnames(data)    
    data['ts'] = pd.to_datetime(data['date']).values.astype(np.int64)//10**9/86400
    data_failure_day = data[data['failure']==1]
#    data_failure_day['ts_failure'] = data_failure_day['ts']
    data = pd.merge(data,data_failure_day[['serial_number','ts']],left_on='serial_number',right_on='serial_number',how='left')
    data['tia'] = data['ts_x']-data['ts_y']+1
    data['tia'] = data['tia'].fillna(0)
    return data[name_meta+['tia']+name_smart]

def select_feature(data):
    data = add_tia(data)
#    stat_ztest = 0
#    scipy.stats.ranksums(x, y)

def kfold_data(datalist_normalized,datalist_select_feature):
    pass

def preprocess_data(path_model,model_name):
    path = path_model+model_name
    data = load_file(path,100000)
    
    [name_meta,name_smart] = get_colnames(data)    
    data = data[data.date >= '2015-01-01']
    
    stat_smart_num_unique = data[name_smart].apply(lambda x:x.nunique())
    data = data[name_meta+stat_smart_num_unique.index[stat_smart_num_unique>1].tolist()]
    data = data.dropna(axis=0,how='any')
    
    data = normalize_data(data)
    
    feature_selected = select_feature(data)
    kfold_data(data,feature_selected)

if __name__=='__main__':
    print '[%s]main start...' %(time.asctime( time.localtime(time.time())))
    
    path_model = os.getenv("HOME")+'/Data/backblaze/model_file/'
    model_names = os.listdir(path_model)
    for mn in model_names[0:1]:
        preprocess_data(path_model,mn)
    
   
    print '[%s]main end...' %(time.asctime( time.localtime(time.time())))
