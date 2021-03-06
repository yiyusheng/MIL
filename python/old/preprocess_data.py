#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 09:34:27 2019

@author: yiyusheng

Preprocess data:
1. Normalize data
2. Add lead time (tia) for instances
3. Select feature by ranksum test
4. Build k-fold group tags for disks

Run for About 25 mins
"""
import os,time,sys
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from scipy.stats import ranksums
from merge_data import load_file,get_colnames 

def select_feature(data):
    [name_meta,name_smart] = get_colnames(data)
    stat_ranksums = pd.DataFrame(list(zip(name_smart,np.zeros(len(name_smart)).tolist())),
                                columns=['name_smart','ranksum'])
    for ns in name_smart:        
        x = data[ns][data['failure']==1]
        y = data[ns][data['failure']==0]
        rs_xp = ranksums(x, y)
        stat_ranksums.loc[stat_ranksums['name_smart']==ns,'ranksum'] = rs_xp[1] #p-value  
    feature_selected = pd.DataFrame(stat_ranksums.loc[stat_ranksums['ranksum']<0.05,'name_smart'],columns=['name_smart'])
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    return [feature_selected,stat_ranksums]

def kfold_data(data,k):
    sn_folds = pd.DataFrame(data['serial_number'].unique(),columns=['serial_number'])
    kf = KFold(n_splits=k,shuffle=True)
    fold_count=0
    for train,valiate in kf.split(sn_folds):
        fold_name = 'fold'+str(fold_count)
        sn_folds[fold_name] = pd.DataFrame(np.zeros(len(sn_folds)).astype(int),columns=[fold_name])
        sn_folds.loc[train,fold_name] = 1    #train=1 valiate=0
        fold_count+=1
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    return sn_folds

def preprocess_data(path_model,model_name,k):
    path = path_model+model_name
    data = load_file(path)
      
    [feature_selected,stat_ranksums] = select_feature(data)
    sn_folds = kfold_data(data,k)
    
    feature_selected.to_csv(path_preprocess+'selected_features_'+model_name,index=0) 
    sn_folds.to_csv(path_preprocess+'sn_folds_'+model_name,index=0)
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    
    return [feature_selected,sn_folds]

if __name__=='__main__':
    print '[%s]main start...' %(time.asctime( time.localtime(time.time())))
    
    path_model = os.getenv("HOME")+'/Data/backblaze/model_file/'
    path_preprocess = os.getenv("HOME")+'/Data/backblaze/model_preprocess/'
    model_names = os.listdir(path_model)
    paras = []
    for mn in model_names:
        paras.append(preprocess_data(path_model,mn,5))
    
   
    print '[%s]main end...' %(time.asctime( time.localtime(time.time())))
