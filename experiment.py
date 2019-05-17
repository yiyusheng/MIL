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

from merge_data import load_file,get_colnames 

def generate_exp_data(data,sn_folds,selected_features,fold_id):
    [name_meta,name_smart] = get_colnames(data)
    name_train_test = name_meta+selected_features['name_smart'].tolist()
    sn_train = sn_folds['serial_number'][sn_folds[fold_id]==1]
    sn_test = sn_folds['serial_number'][sn_folds[fold_id]==0]
    train = data[data['serial_number'].isin(sn_train),name_train_test]
    test = data[data['serial_number'].isin(sn_test),name_train_test]
    
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    return [train,test]

def build_model_mil(train):
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    return 0

def predict_model_mil(model_mil,test):
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    return 0

def evaluate_model_mil(predicted_result_mil):
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    return 0

def experiment(path_model,path_preprocess,mn):
    
    data = load_file(path_model+mn,nrows=1e5)
    selected_features = load_file(path_preprocess+'selected_features_'+mn)
    sn_folds = load_file(path_preprocess+'sn_folds_'+mn)
    
    for i in range(1,6):
        fold_id = 'fold'+str(i)
        [train,test] = generate_exp_data(data,sn_folds,selected_features,fold_id)
        model_mil = build_model_mil(train)
        predicted_result_mil = predict_model_mil(model_mil,test)
        perf_mil = evaluate_model_mil(predicted_result_mil)
    
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    return perf_mil

if __name__=='__main__':
    print '[%s]%s start...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)    
    
    path_model = os.getenv("HOME")+'/Data/backblaze/model_file/'
    path_preprocess = os.getenv("HOME")+'/Data/backblaze/model_preprocess/'
    model_names = os.listdir(path_model)
    
    paras = []
    for mn in model_names:
        paras.append(experiment(path_model,path_preprocess,mn))
        print '[%s]model %s done...\n' %(time.asctime(time.localtime(time.time())),mn)
    
   
    print '[%s]%s end...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)