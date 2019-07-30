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
import os,time,sys
import numpy as np
import pandas as pd
import misvm

from merge_data import load_file,get_colnames
from datetime import datetime
from sklearn.metrics import confusion_matrix


def load_files(path_model,path_preprocess,mn,nrows=-1):
    start_time = datetime.now()    
    if nrows==-1:
        data = load_file(path_model+mn)
    else:
        data = load_file(path_model+mn,nrows=nrows)
    selected_features = load_file(path_preprocess+'selected_features_'+mn)
    sn_folds = load_file(path_preprocess+'sn_folds_'+mn)
    print '[%s]%s done for %d seconds...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    return [data,selected_features,sn_folds]

def deduplicate_data(data):
    start_time = datetime.now()
    [name_meta,name_smart] = get_colnames(data)
    name_smart.remove('smart_9_raw')
    data = data.drop_duplicates(['serial_number']+name_smart,keep='last')
     
    print '[%s]%s done for %d seconds...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    return data

def select_instant(data,num_pos_bags,num_neg_bags):
    start_time = datetime.now()
    data_pos = data.loc[(data['tia']>=1) & (data['tia']<=num_pos_bags)].reset_index(drop=True)
    data_pos = data_pos.sort_values(['serial_number','date'])
    
    fn = lambda df:df.loc[np.random.choice(df.index,min(len(df.index),num_neg_bags),False)]
    data_neg = data.loc[data['tia']==0]
    data_neg = data_neg.groupby('serial_number').apply(fn).reset_index(drop=True)
    data_neg = data_neg.sort_values(['serial_number','date'])
    
    data = pd.concat([data_pos,data_neg]).reset_index(drop=True)
    
    print '[%s]%s done for %d seconds...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    return data

def generate_exp_data(data,sn_folds,selected_features,fold_id):
    start_time = datetime.now()
    [name_meta,name_smart] = get_colnames(data)
    sf = selected_features['name_smart'].tolist()
    name_train_test = name_meta+sf
    sn_train = sn_folds['serial_number'][sn_folds[fold_id]==1]
    sn_test = sn_folds['serial_number'][sn_folds[fold_id]==0]
    train = data[data['serial_number'].isin(sn_train)][name_train_test]
    test = data[data['serial_number'].isin(sn_test)][name_train_test]
    [train_bags,train_labels] = generate_bags(train,sf)
    [test_bags,test_labels] = generate_bags(test,sf)
    
    print '[%s]%s done for %d seconds...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    return [train_bags,train_labels,test_bags,test_labels]

def generate_bags(train,sf):
    gb = train.groupby(train['serial_number'])
    bags = [gb.get_group(x)[sf].values for x in gb.groups]
    labels = gb['failure'].apply(max)
    return [bags,labels]

def generate_balance_index(train_labels,times_pos=1):
    index_pos = train_labels.index[train_labels==1]
    
    index_neg = train_labels.index[train_labels==0]
    len_neg = len(index_neg)
    len_need_neg = int(len(index_pos)*times_pos)
    index_neg_balance = index_neg[np.random.choice(range(len_neg),len_need_neg,replace=False)]
    
    return pd.Index.union(index_pos,index_neg_balance)

def model_mil(train_bags,train_labels,test_bags,test_labels):
    start_time = datetime.now()
    classifiers = {}
    classifiers['MissSVM'] = misvm.MissSVM(kernel='linear', C=1.0, max_iters=20)
    classifiers['sbMIL'] = misvm.sbMIL(kernel='linear', eta=0.1, C=1e2) 
    classifiers['SIL'] = misvm.SIL(kernel='linear', C=1.0)

    # Train/Evaluate classifiers
    perf = {}
    for algorithm, classifier in classifiers.items():
        classifier.fit(train_bags, train_labels)
        predictions = classifier.predict(test_bags)
        if algorithm == 'sbMIL' or algorithm == 'MissSVM':
            predictions = np.sign(predictions)+1
        [[TN,FP],[FN,TP]] = confusion_matrix(test_labels,predictions)
        FDR = TP/(TP+FN)
        FAR = FP/(TN+FP)
        perf[algorithm] = {'FDR':FDR,'FAR':FAR} 
        print '[%s]algorithm %s done...' %(time.asctime(time.localtime(time.time())),algorithm)
    print '[%s]%s done for %d seconds...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    return perf

if __name__=='__main__':
    print '[%s]%s start...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)    
    
    path_model = os.getenv("HOME")+'/Data/backblaze/model_file/'
    path_preprocess = os.getenv("HOME")+'/Data/backblaze/model_preprocess/'
    model_names = os.listdir(path_model)
    model_names = model_names[1:2] #MODIFY
    
    perf_model = {}
    for mn in model_names: 
        [data,selected_features,sn_folds]=load_files(path_model,path_preprocess,mn)
        data = deduplicate_data(data)
        data = select_instant(data,30,5)
        perf_mil = {}
        for i in range(1): #MODIFY one training-testing set
            fold_id = 'fold'+str(i)
            [train_bags,train_labels,test_bags,test_labels] = generate_exp_data(data,sn_folds,selected_features,fold_id)
            idx_train = generate_balance_index(train_labels)
            perf_mil[fold_id] = model_mil(train_bags.loc[idx_train],train_labels[idx_train],test_bags,test_labels) 
        perf_model[mn]=perf_mil 
        print perf_mil
   
    print '[%s]%s end...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
