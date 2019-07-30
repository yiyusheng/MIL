#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:17:42 2019

@author: yiyusheng

filter data
group data by models
statistic data by models
"""

import time,sys
import pandas as pd
import numpy as np
from base_func import get_colnames
from scipy.stats import ranksums
from sklearn.model_selection import KFold


# %% filter and save
def filter_data(data,save=1,start_date='2015-01-01'):
    [name_meta,name_smart] = get_colnames(data)
    stat_null = data[name_smart].isnull().sum()/len(data)*100
    name_smart_valid = stat_null.index[stat_null<=10].values
    data_filter = data[name_meta+name_smart_valid.tolist()]
    
    data_filter = data_filter[data_filter.date >= start_date]
    
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)

    return data_filter

def add_tia(data):
    [name_meta,name_smart] = get_colnames(data)    
    data['ts'] = pd.to_datetime(data['date']).values.astype(np.int64)//10**9/86400
    idx = data[data['failure']==1].groupby('serial_number')['date'].transform(max) == data[data['failure']==1]['date']
    data_failure_day = data[data['failure']==1][idx]
#    data_failure_day = data[data['failure']==1]
    data = pd.merge(data,data_failure_day[['serial_number','ts']],left_on='serial_number',right_on='serial_number',how='left')
    data['tia'] = data['ts_y']-data['ts_x']+1
    data['tia'] = data['tia'].fillna(0)
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)

    return data[name_meta+['tia']+name_smart]

def normalize_data(data):
    [name_meta,name_smart] = get_colnames(data)    
    data[name_smart]=data[name_smart].apply(lambda x:(x-x.min())/(x.max()-x.min()))
    data[name_smart]=data[name_smart].fillna(0)
    data[name_smart] = data[name_smart].round(6)
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    return data

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



if __name__=='__main__':
    pass
    
