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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from para import load_files,set_paras
from datetime import datetime

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid

def evaluate_model(test_label,test_pred,test_meta):
    test_meta['y_test'] = test_label
    test_meta['y_pred'] = test_pred
    gb = test_meta.groupby('serial_number')
    disk_test = gb['y_test'].apply(lambda x: sum(x==1)>0)
    disk_pred = gb['y_pred'].apply(lambda x: sum(x==1)>0)
    serial_number = gb.groups.keys()
    
    disk_result = pd.DataFrame({'serial_number':serial_number,
                                'disk_test':disk_test,
                                'disk_pred':disk_pred})
    instant_result = test_meta
    
    [[TN,FP],[FN,TP]] = confusion_matrix(disk_test,disk_pred)
    FDR = TP+FN==0 and 0 or TP/float(TP+FN)
    FAR = TN+FP==0 and 0 or FP/float(TN+FP)
    
    return [{'FDR':FDR,'FAR':FAR},disk_result,instant_result] 

def model_svm(dataexp,quiet=1):
    start_time = datetime.now()    
    [train,train_meta,train_label,test,test_meta,test_label] = dataexp
    model = SVC(gamma='auto')
    model.fit(train, train_label)
    test_pred = model.predict(test)
    
    m = evaluate_model(test_label,test_pred,test_meta)
    print '%s done for %d seconds...' %(sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    return m[0]

def model_rf(dataexp,quiet=1):
    start_time = datetime.now()    
    [train,train_meta,train_label,test,test_meta,test_label] = dataexp
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train, train_label)
    test_pred = model.predict(test)
 
    m = evaluate_model(test_label,test_pred,test_meta)
    print '%s done for %d seconds...' %(sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    return m[0]

def model_nb(dataexp,quiet=1):
    start_time = datetime.now()    
    [train,train_meta,train_label,test,test_meta,test_label] = dataexp  
    model = GaussianNB()
    model.fit(train, train_label)
    test_pred = model.predict(test)
 
    m = evaluate_model(test_label,test_pred,test_meta)
    print '%s done for %d seconds...' %(sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    return m[0]

def stat_number(dataexp,data_name,paras):
    [train,train_meta,train_label,test,test_meta,test_label] = dataexp 
    [f_id,num_neg,rate_neg,time_window] = paras    
    train_label_disk = train_meta.groupby('serial_number')['failure'].apply(max)
    test_label_disk = test_meta.groupby('serial_number')['failure'].apply(max)

    [tr_in,tr_ip] = np.unique(train_label,return_counts=True)[1]
    [te_in,te_ip] = np.unique(test_label,return_counts=True)[1]
    [tr_dn,tr_dp] = np.unique(train_label_disk,return_counts=True)[1]
    [te_dn,te_dp] = np.unique(test_label_disk,return_counts=True)[1]
    
    print 'Data %s start...' %(data_name)
    print 'f_id:%s num_neg:%d rate_neg:%d time_window:%d' %(f_id,num_neg,rate_neg,time_window)
    print 'tr_dn:%d tr_dp:%d te_dn:%d te_dp:%d' %(tr_dn,tr_dp,te_dn,te_dp)
    print 'tr_in:%d tr_ip:%d te_in:%d te_ip:%d' %(tr_in,tr_ip,te_in,te_ip)
    print 'tr_d:%d te_d:%d tr_dni:%.2f tr_dpi:%.2f te_dni:%.2f te_dpi:%.2f ' \
    %(tr_dn+tr_dp,te_dn+te_dp,tr_in/float(tr_dn),tr_ip/float(tr_dp),
      te_in/float(te_dn),te_ip/float(te_dp))
  
def exp_data(paras_grid,datas,data_name,upper=2):
    result_svm = []
    result_rf = []
    result_nb = []
    if upper==-1:
        upper = len(paras_grid)
    idx = range(upper)
    for i in idx:
        paras = paras_grid.loc[i].tolist()
        dataexp = set_paras(datas,paras,1)
        stat_number(dataexp,data_name,paras)
        r_svm = model_svm(dataexp)
        r_rf = model_rf(dataexp)
        r_nb = model_nb(dataexp)
        result_svm.append(r_svm)
        result_rf.append(r_rf)
        result_nb.append(r_nb)
        print 'FDR_svm:%.3f FAR_svm:%.3f FDR_rf:%.3f FAR_rf:%.3f FDR_nb:%.3f FAR_nb:%.3f'\
        %(r_svm['FDR'],r_svm['FAR'],r_rf['FDR'],r_rf['FAR'],r_nb['FDR'],r_nb['FAR'])
        print '[%s]paras[%d] for dataset %s end...\n' %(time.asctime(time.localtime(time.time())),i,data_name)   
    
    result_svm = pd.DataFrame(result_svm)
    result_rf = pd.DataFrame(result_rf)
    result_nb = pd.DataFrame(result_nb)
    result_svm.columns = ['FAR_svm','FDR_svm']
    result_rf.columns = ['FAR_rf','FDR_rf']
    result_nb.columns = ['FAR_nb','FDR_nb']
    
    result = pd.concat([paras_grid[0:upper],result_svm,result_rf,result_nb],axis=1)
    result['data_name'] = data_name
    return result

def visualize_result(result,model_name):
    x_label = 'FAR_'+model_name
    y_label = 'FDR_'+model_name
    groups = result.groupby('data_name')
    
    fig,ax=plt.subplots()
    ax.margins(0.05)
    for name,group in groups:
        ax.plot(group[x_label],group[y_label],marker='o',linestyle='',ms=12,label=name)
    ax.legend()
    
    plt.show()
    return plt
    
if __name__=='__main__':
    # config
    path_preprocess_bb = os.getenv("HOME")+'/Data/backblaze/model_preprocess/' 
    path_preprocess_baidu = os.getenv("HOME")+'/Data/baidu/' 
    path_preprocess_murry = os.getenv("HOME")+'/Data/murry05/' 
    mn_bb = 'ST4000DM000'
    mn_baidu = 'baidu'
    mn_murry = 'murry'
    
    # Set paras
    paras = [30,5,2,0]
    paras_grid = pd.DataFrame(list(ParameterGrid({'f_id':[0],
                                    'num_neg':[-1],
                                    'rate_neg':[1,2,4,8,16,32,64],
                                    'time_window':[7,14]})))
    
    # Load data and execute experiment  
    datas_murry = load_files(path_preprocess_murry,mn_murry)
    result_murry = exp_data(paras_grid,datas_murry,'murry',-1)
#    datas_bb = load_files(path_preprocess_bb,mn_bb)
#    result_bb = exp_data(paras_grid,datas_bb,'backblaze',-1)  
#    datas_baidu = load_files(path_preprocess_baidu,mn_baidu)
#    result_baidu = exp_data(paras_grid,datas_baidu,'baidu',2)
    
    
    # Experiment   
#    result = result_baidu
#    result = result.append(result_baidu)
#    result = result.append(result_murry)
#    
#    # Visualization
#    fig_svm = visualize_result(result,'svm')
#    fig_rf = visualize_result(result,'rf')
#    fig_nb = visualize_result(result,'nb')
    
    
    print '[%s]%s end...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)