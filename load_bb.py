#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:52:13 2019

@author: yiyusheng
"""

import os,time,sys
from filter import filter_data,add_tia,normalize_data,select_feature,kfold_data
from base_func import load_file

def preprocess_data(path_model,model_name,k,path_preprocess):
    path = path_model+model_name
    
    data = load_file(path)
    data = filter_data(data)   
    data = add_tia(data)
    data = normalize_data(data)
    data.to_csv(path_preprocess+'preprocessed_data_'+model_name,index=0)
      
    [feature_selected,stat_ranksums] = select_feature(data)
    feature_selected.to_csv(path_preprocess+'selected_features_'+model_name,index=0) 

    sn_folds = kfold_data(data,k)   
    sn_folds.to_csv(path_preprocess+'sn_folds_'+model_name,index=0)
    
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    return [feature_selected,sn_folds]

if __name__=='__main__':
    print '[%s]main start...' %(time.asctime( time.localtime(time.time())))
    
    path_preprocess = os.getenv("HOME")+'/Data/backblaze/model_preprocess/'
    path_model = os.getenv("HOME")+'/Data/backblaze/model_file/'
    model_names = os.listdir(path_model)
    
    paras = []
    for mn in model_names:
        paras.append(preprocess_data(path_model,mn,5,path_preprocess))
    
    print '[%s]main end...' %(time.asctime( time.localtime(time.time())))