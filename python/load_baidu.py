#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:52:13 2019

@author: yiyusheng
"""

import os,time,datetime,sys
import pandas as pd
#import numpy as np
import filter
from base_func import get_colnames

def add_date(data):
    data['date'] = datetime.datetime.strptime('2013-01-01','%Y-%m-%d')
    date_cum = data.groupby('serial_number').cumcount()
    data['date'] = data['date'] + pd.to_timedelta(date_cum,unit='D')
    
    [name_meta,name_smart] = get_colnames(data)
    data = data[name_meta+name_smart]
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    return data

if __name__=='__main__':
    print '[%s]main start...' %(time.asctime( time.localtime(time.time())))
    model_name = 'baidu'
    k=5
    path_load = os.getenv("HOME")+'/Data/baidu/'
    
    # Load
    data = pd.read_csv(path_load+'Disk_SMART_dataset.txt',sep=',',header=None)
    data.columns = ['serial_number','failure','smart_1_normalized','smart_3_normalized',
                    'smart_5_normalized','smart_7_normalized','smart_9_normalized',
                    'smart_187_normalized','smart_189_normalized','smart_194_normalized',
                    'smart_195_normalized','smart_197_normalized','smart_5_raw',
                    'smart_197_raw',]
    
    # Preprocess data
    data['failure']=(data['failure']*(-1)+1)/2
    data = add_date(data)
    data = filter.filter_data(data,save = 1,start_date='2010-01-01')   
    data = filter.add_tia(data)
    data = filter.normalize_data(data)
    data.to_csv(path_load+'preprocessed_data_'+model_name,index=0)
    
    # Select features
    [feature_selected,stat_ranksums] = filter.select_feature(data)
    feature_selected.to_csv(path_load+'selected_features_'+model_name,index=0) 

    # Build k-fold tags
    sn_folds = filter.kfold_data(data,k)   
    sn_folds.to_csv(path_load+'sn_folds_'+model_name,index=0)