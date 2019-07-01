#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:21:49 2019

@author: yiyusheng

NOTE: DO NOT USE THIS SCRIPT since MIL package is written by matlab
Experiment:
    1. generate training set and testing set
    2. build MIL model on training set
    3. predict the result for validating set
    4. evaluate the predicted result
    
"""
import time,sys
import numpy as np
import pandas as pd

from base_func import load_file,get_colnames
from datetime import datetime


def load_files(path_preprocess,mn,nrows=-1,quiet=0):
    start_time = datetime.now()    
    if nrows==-1:
        data = load_file(path_preprocess+'preprocessed_data_'+mn)
    else:
        data = load_file(path_preprocess+'preprocessed_data_'+mn,nrows=nrows)
    selected_features = load_file(path_preprocess+'selected_features_'+mn)
    sn_folds = load_file(path_preprocess+'sn_folds_'+mn)
    if not quiet:
        print '[%s]%s done for %d seconds...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    return [data,selected_features,sn_folds]

def select_instant(data,time_window,num_neg,quiet=0):
    start_time = datetime.now()
    data_pos = data.loc[(data['tia']>=1) & (data['tia']<=time_window)].reset_index(drop=True)
    data_pos = data_pos.sort_values(['serial_number','date'])
        
    data_neg = data.loc[data['tia']==0]
    if not num_neg==-1:
        fn = lambda df:df.loc[np.random.choice(df.index,min(len(df.index),num_neg),False)]
        data_neg = data_neg.groupby('serial_number').apply(fn).reset_index(drop=True)
        data_neg = data_neg.sort_values(['serial_number','date'])
    
    data = pd.concat([data_pos,data_neg]).reset_index(drop=True)
    
    if not quiet:
        print '[%s]%s done for %d seconds...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    return data

def generate_train_test(data,sn_folds,selected_features,fold_id,time_window,num_neg,quiet=0):
    start_time = datetime.now()
    [name_meta,name_smart] = get_colnames(data)
    sf = selected_features['name_smart'].tolist()
    name_train_test = sf
    
    sn_train = sn_folds['serial_number'][sn_folds[fold_id]==1]
    train = data[data['serial_number'].isin(sn_train)]
    train = select_instant(train,time_window,num_neg,quiet)
    train_meta = train[name_meta]
    train_label = (train_meta['tia']>0).astype(int)
    train = train[name_train_test] 
    
    sn_test = sn_folds['serial_number'][sn_folds[fold_id]==0]    
    test = data[data['serial_number'].isin(sn_test)][name_train_test]
    test_meta = data[data['serial_number'].isin(sn_test)][name_meta]
    test_label = (test_meta['tia']>0).astype(int)
    
    if not quiet:
        print '[%s]%s done for %d seconds...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    return [train,train_meta,train_label,
            test,test_meta,test_label]

def generate_balance_index(train_label,rate_neg=1):
    index_pos = train_label.index[train_label==1]
    
    index_neg = train_label.index[train_label==0]
    len_neg = len(index_neg)
    len_need_neg = min(len_neg,int(len(index_pos)*rate_neg))
    index_neg_balance = index_neg[np.random.choice(range(len_neg),len_need_neg,replace=False)]
    
    return pd.Index.union(index_pos,index_neg_balance)

def set_paras(datas,f_id,num_neg,rate_neg,time_window,quiet=1):

    [data,selected_features,sn_folds]=datas
#    data = select_instant(data,time_window,num_neg,quiet,)
    fold_id = 'fold'+str(f_id)
    [train,train_meta,train_label,test,test_meta,test_label] = generate_train_test(data,sn_folds,selected_features,fold_id,time_window,num_neg,quiet)
    idx_train = generate_balance_index(train_label,rate_neg)
    idx_train = idx_train.values
    return [train.loc[idx_train,:],train_meta.loc[idx_train],train_label.loc[idx_train],
            test,test_meta,test_label]


if __name__=='__main__':
    pass