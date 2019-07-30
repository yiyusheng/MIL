#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:52:13 2019

@author: yiyusheng
"""

import os,time
import pandas as pd
import filter
from load_baidu import add_date


if __name__=='__main__':
    print '[%s]main start...' %(time.asctime( time.localtime(time.time())))
    model_name = 'murry'
    k=5
    path_load = os.getenv("HOME")+'/Data/murry05/'
    
    # Load
    data = pd.read_csv(path_load+'harddrive1.arff',sep=',',header=None,skiprows=95)
    columns = pd.read_csv(path_load+'murry_columns',sep=',',header=None)
    data.columns = columns[0].values
    
    # Preprocess data
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