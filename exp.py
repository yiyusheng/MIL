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
from sklearn.model_selection import ParameterGrid,GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import make_scorer

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
    
    return {'FDR':round(FDR,3),'FAR':round(FAR,3)}


def model_svm(dataexp,paras_model):
    start_time = datetime.now()    
    [train,train_meta,train_label,test,test_meta,test_label] = dataexp
    m=[]
    for p in paras_model:
        model = SVC(**p)
        model.fit(train, train_label)
        test_pred = model.predict(test)   
        p.update(evaluate_model(test_label,test_pred,test_meta))
        m.append(p)
    print '%s done for %ds...' %(sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    m = pd.DataFrame(m)
    m = m[['FDR','FAR','C','gamma']]
    return m

def model_rf(dataexp,paras_model):
    start_time = datetime.now()    
    [train,train_meta,train_label,test,test_meta,test_label] = dataexp
    m=[]
    for p in paras_model:
        model = RandomForestClassifier(min_samples_leaf=20,max_features='sqrt',random_state=10,
                                       n_estimators=100,n_jobs=10,**p)
        model.fit(train, train_label)
        test_pred = model.predict(test)   
        p.update(evaluate_model(test_label,test_pred,test_meta))
        m.append(p)
    print '%s done for %ds...' %(sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    m = pd.DataFrame(m)
    m = m[['FDR','FAR','max_depth','min_samples_split']]
    return m

def model_nb(dataexp,paras_model):
    start_time = datetime.now()    
    [train,train_meta,train_label,test,test_meta,test_label] = dataexp
    m=[]
    for p in paras_model:
        model = GaussianNB(**p)
        model.fit(train, train_label)
        test_pred = model.predict(test)   
        p.update(evaluate_model(test_label,test_pred,test_meta))
        m.append(p.values())
    print '%s done for %ds...' %(sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
    m = pd.DataFrame(m)
    m = m[['FDR','FAR','p1','p2']]
    return m



def stat_number(dataexp,data_name,f_id,num_neg,rate_neg,time_window):
    [train,train_meta,train_label,test,test_meta,test_label] = dataexp 
    train_label_disk = train_meta.groupby('serial_number')['failure'].apply(max)
    test_label_disk = test_meta.groupby('serial_number')['failure'].apply(max)

    [tr_in,tr_ip] = np.unique(train_label,return_counts=True)[1]
    [te_in,te_ip] = np.unique(test_label,return_counts=True)[1]
    [tr_dn,tr_dp] = np.unique(train_label_disk,return_counts=True)[1]
    [te_dn,te_dp] = np.unique(test_label_disk,return_counts=True)[1]
    
    print '\nData %s start with f_id:%s num_neg:%d rate_neg:%.2f time_window:%d...' %(data_name,f_id,num_neg,rate_neg,time_window)
    
#    print 'tr_dn:%d tr_dp:%d te_dn:%d te_dp:%d' %(tr_dn,tr_dp,te_dn,te_dp)
#    print 'tr_in:%d tr_ip:%d te_in:%d te_ip:%d' %(tr_in,tr_ip,te_in,te_ip)
#    print 'tr_d:%d te_d:%d tr_dni:%.2f tr_dpi:%.2f te_dni:%.2f te_dpi:%.2f ' \
#    %(tr_dn+tr_dp,te_dn+te_dp,tr_in/float(tr_dn),tr_ip/float(tr_dp),
#      te_in/float(te_dn),te_ip/float(te_dp))
  
def exp_data(paras_set,datas,data_name):
    start_time = datetime.now()    
    
    [paras_data,paras_svm,paras_rf,paras_nb] = paras_set              
    result_svm = []
    result_rf = []
    result_nb = []
    
    for p in paras_data:
        dataexp = set_paras(datas = datas,**p)
        stat_number(dataexp,data_name,**p)
        
        r_svm = model_svm(dataexp,paras_svm)
        r_rf = model_rf(dataexp,paras_rf)
        r_nb = r_rf
        
        r_svm[p.keys()] = pd.DataFrame([p.values()],index=r_svm.index)
        r_rf[p.keys()] = pd.DataFrame([p.values()],index=r_rf.index)
        r_nb[p.keys()] = pd.DataFrame([p.values()],index=r_nb.index) 
        
        result_svm.append(r_svm)
        result_rf.append(r_rf)
        result_nb.append(r_nb)
    
    result_svm = pd.concat(result_svm); result_svm['alg'] = 'svm'
    result_rf = pd.concat(result_rf); result_rf['alg'] = 'rf'
    result_nb = pd.concat(result_nb); result_nb['alg'] = 'nb'
    
    col_plot = ['FDR','FAR','alg']
    result = pd.concat([result_svm[col_plot],result_rf[col_plot],result_nb[col_plot]])
    result['data_name'] = data_name
    print '[%s]\n%s dataset %s done for %ds...' %(time.asctime(time.localtime(time.time())),
           sys._getframe().f_code.co_name,data_name,(datetime.now()-start_time).seconds)
    return result

def visualize_result(result,by='alg'):
    result = result.sort_values(['alg','FAR'])
    groups = result.groupby(by)

    fig,ax=plt.subplots()
    ax.margins(0.05)
    for name,group in groups:
        ax.plot(group['FAR'],group['FDR'],marker='o',linestyle='-',ms=12,label=name)
    ax.legend()
    
    plt.show()
      
if __name__=='__main__':
    # config
    path_preprocess_bb = os.getenv("HOME")+'/Data/backblaze/model_preprocess/' 
    path_preprocess_baidu = os.getenv("HOME")+'/Data/baidu/' 
    path_preprocess_murry = os.getenv("HOME")+'/Data/murry05/' 
    mn_bb = 'ST4000DM000'
    mn_baidu = 'baidu'
    mn_murry = 'murry'
    
    # Set paras
#    paras_data = ParameterGrid({'f_id':[0],'num_neg':[-1],'rate_neg':[1],'time_window':[3,7]})
    paras_data = ParameterGrid({'f_id':[0],'num_neg':[-1],'rate_neg':[0.5,1,4,16],'time_window':[7,14,28]})
    paras_svm = ParameterGrid({'C':[0.001,0.01,0.1,1],'gamma':[0.01,0.1,1]})
    paras_rf = ParameterGrid({'max_depth':range(3,15,3),'min_samples_split':[0.001,0.01,0.1,0.2]})
    paras_nb = ParameterGrid({'priors_0':[x/float(10) for x in range(1,10)],'priors_1':[x/float(10) for x in range(9,0,-1)]})
    paras_set = [paras_data,paras_svm,paras_rf,paras_nb]
    
    # Load data and execute experiment  
    datas_murry = load_files(path_preprocess_murry,mn_murry);datas=datas_murry;data_name='murry';upper=-1
    result_murry = exp_data(paras_set,datas_murry,'murry')   
    datas_bb = load_files(path_preprocess_bb,mn_bb)
    result_bb = exp_data(paras_set,datas_bb,'backblaze')  
    datas_baidu = load_files(path_preprocess_baidu,mn_baidu)
    result_baidu = exp_data(paras_set,datas_baidu,'baidu')    
    result_all = pd.concat([result_murry,result_baidu,result_bb])
    
  
#    # Visualization
    fig_svm = visualize_result(result_all[result_all['alg']=='svm'],'data_name')
    fig_rf = visualize_result(result_all[result_all['alg']=='rf'],'data_name')
    fig_nb = visualize_result(result_all[result_all['alg']=='nb'],'data_name')
    
    
    print '[%s]%s end...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)


#%% Noneed
    #def model_gridsearch(datas,paras_data,paras_model):
##    stat_number(dataexp,data_name,**p)  
#    def custom_fpr(ground_truth,predictions):
#        fpr,tpr,_=roc_curve(ground_truth,predictions,pos_label=1)
#        print fpr,tpr
#        return fpr
#    my_fpr = make_scorer(custom_fpr)
#    
#    paras_model = {'max_depth':range(3,9,3),'min_samples_split':[0.01]}
#    r_set=[]
#    for pdt in paras_data:
#        [train,train_meta,train_label,test,test_meta,test_label] = set_paras(datas = datas,**pdt)
#        gs = GridSearchCV(estimator = RandomForestClassifier(min_samples_leaf=20, max_features='sqrt' ,random_state=10,n_estimators=100),
#                          param_grid = paras_model, scoring=my_fpr,cv=3)
#        gs.fit(train,train_label)    
#        
#        r = pd.DataFrame(gs.cv_results_['params'])
#        r[pdt.keys()] = pd.DataFrame([pdt.values()],index=r.index)
#        r['mean_test_recall'] = gs.cv_results_['mean_test_recall']
#        r['mean_test_precision'] = gs.cv_results_['mean_test_precision']  
#        r['mean_test_roc_auc'] = gs.cv_results_['mean_test_roc_auc']  
#        
#        r_set.append(r)
#        
#    r_all = pd.concat(r_set)
#    return r_all
#    
#def model_rf_gs(datas,f_id,num_neg,rate_neg,time_window,max_depth,min_samples_split):
#    [train,train_meta,train_label,test,test_meta,test_label] = set_paras(datas,f_id,num_neg,rate_neg,time_window,quiet=1)
#    model = RandomForestClassifier(max_depth = 30,min_samples_split = 20,min_samples_leaf=20, 
#                                   max_features='sqrt' ,random_state=10,n_estimators=100,n_jobs=10)
#    model.fit(train,train_label)
#    
#    test_pred = model.predict(test) 
#    fpr,tpr = roc_curve(test_label,test_pred)
#    score = auc(fpr,tpr)
#    return score
#
#
#def model_rf(dataexp,paras_rf):
#    start_time = datetime.now()    
#  
#    [train,train_meta,train_label,test,test_meta,test_label] = dataexp
#    m=[]
#    for p in paras_rf:
#        model = RandomForestClassifier(n_estimators=100,n_jobs=10,**p)
#        model.fit(train, train_label)
#        test_pred = model.predict(test)   
#        p.update(evaluate_model(test_label,test_pred,test_meta))
#        m.append(p.values())
#    print '%s done for %ds...' %(sys._getframe().f_code.co_name,(datetime.now()-start_time).seconds)
#    m = pd.DataFrame(m)
#    print 'result:\n %s' %(m)
#    return m