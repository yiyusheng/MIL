#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:17:42 2019

@author: yiyusheng

filter data
group data by models
statistic data by models
"""

import os,time,sys
import pandas as pd
import numpy as np
from merge_data import load_file,get_colnames


# %% filter and save
def filter_data(data,save=1,col_meta=5,start_date='2015-01-01'):
    data_names = data.columns.values
    data_names_meta = data_names[0:col_meta]
    data_names_attr = data_names[(col_meta+1):len(data_names)]
    stat_null = data[data_names_attr].isnull().sum()/len(data)*100
    valid_names = stat_null.index[stat_null<=10].values
    data_filter = data[data_names_meta.tolist()+valid_names.tolist()]
    data_filter = data_filter[data_filter.date >= start_date]
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)

    return data_filter

def add_tia(data):
    [name_meta,name_smart] = get_colnames(data)    
    data['ts'] = pd.to_datetime(data['date']).values.astype(np.int64)//10**9/86400
    data_failure_day = data[data['failure']==1]
    data = pd.merge(data,data_failure_day[['serial_number','ts']],left_on='serial_number',right_on='serial_number',how='left')
    data['tia'] = data['ts_y']-data['ts_x']+1
    data['tia'] = data['tia'].fillna(0)
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)

    return data[name_meta+['tia']+name_smart]

def normalize_data(data):
    [name_meta,name_smart] = get_colnames(data)    
    data[name_smart]=data[name_smart].apply(lambda x:(x-x.min())/(x.max()-x.min()))
    data[name_smart]=data[name_smart].fillna(0)
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    return data

def group_data_by_models(data_filter,path_model,save=1):
    model_count = data_filter['model'].value_counts()/len(data_filter)
    valid_model = model_count[0:6]
    data_filter_major_models = data_filter[data_filter['model'].isin(valid_model.index.values)]
    
    data_models = []
    for m in valid_model.index.values:
        data_model = data_filter_major_models[data_filter_major_models['model']==m]
        [name_meta,name_smart] = get_colnames(data_model) 
        stat_smart_num_unique = data_model[name_smart].apply(lambda x:x.nunique())
        data_model = data_model[name_meta+stat_smart_num_unique.index[stat_smart_num_unique>1].tolist()]
        data_model = data_model.dropna(axis=0,how='any')
        data_model = normalize_data(data_model)
        data_models.append(data_model)
        if save==1:
            data_model.to_csv(path_model+m,index=0)        
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    return data_models

def stat_by_models(data_filter,path_stat,save=1):
#    stat_model_num_item = data_filter['model'].value_counts()
#    stat_model_num_failure = data_filter[data_filter['failure']==1]['model'].value_counts()
    
    stat_model_num_item = data_filter.groupby(['model']).size().to_frame('item')
    stat_model_num_sn = data_filter.groupby(['model'])['serial_number'].nunique().to_frame()
    stat_model_num_failure = data_filter[data_filter['failure']==1].groupby(['model']).size().to_frame('failure')
    stat_model_date_num_item = data_filter.groupby(['model','date']).size().to_frame('item').reset_index()
    stat_model_date_num_failure = data_filter[data_filter['failure']==1].groupby(['model','date']).size().to_frame('failure').reset_index()
    stat_model_count = (data_filter['model'].value_counts()/len(data_filter)).to_frame()
    
    if save==1:
        stat_model_num_item.to_csv(path_stat+'stat_model_num_item')  
        stat_model_num_sn.to_csv(path_stat+'stat_model_num_sn')  
        stat_model_num_failure.to_csv(path_stat+'stat_model_num_failure')  
        stat_model_date_num_item.to_csv(path_stat+'stat_model_date_num_item',index=0)  
        stat_model_date_num_failure.to_csv(path_stat+'stat_model_date_num_failure',index=0)  
        stat_model_count.to_csv(path_stat+'stat_model_count')
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)    
    return [stat_model_num_item,stat_model_num_sn,stat_model_num_failure,stat_model_date_num_item,stat_model_date_num_failure,stat_model_count]    

if __name__=='__main__':
    print '[%s]main start...' %(time.asctime( time.localtime(time.time())))
    
    path_load = os.getenv("HOME")+'/Data/backblaze/data_all'
    path_model = os.getenv("HOME")+'/Data/backblaze/model_file/'
    path_stat =  os.getenv("HOME")+'/Data/backblaze/stat/'    
    path_year = os.getenv("HOME")+'/Data/backblaze/year_file/'
    
    data15 = load_file(path_year+'data_2015')
    data16 = load_file(path_year+'data_2016')
    data17 = load_file(path_year+'data_2017')
    data18 = load_file(path_year+'data_2018')
    data19 = load_file(path_year+'data_2019')
    data_list = [data15,data16,data17,data18,data19]
    data_all = pd.concat(data_list,sort=False)
    
#    data_filter = filter_data(data_all)   
#    data_filter = add_tia(data_filter)
#    [m1,m2,m3,m4,m5,m6] = group_data_by_models(data_filter,path_model)
#    [s1,s2,s3,s4,s5,s6] = stat_by_models(data_filter,path_stat)
    
    print '[%s]main end...' %(time.asctime( time.localtime(time.time())))
