#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:17:42 2019

@author: yiyusheng

filter data
group data by models
statistic data by models
"""

import os,time
from merge_data import load_file,get_colnames

# %% filter and save
def filter_data(data,save=1):
    data_names = data.columns.values
    data_names_meta = data_names[0:5]
    data_names_attr = data_names[6:len(data_names)]
    stat_null = data[data_names_attr].isnull().sum()/len(data)*100
    valid_names = stat_null.index[stat_null<=10].values
    data_filter = data[data_names_meta.tolist()+valid_names.tolist()]
    data_filter = data_filter[data_filter.date >= '2015-01-01']

    return data_filter

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
        data_models.append(data_model)
        if save==1:
            data_model.to_csv(path_model+m,index=0)        
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
    
    return [stat_model_num_item,stat_model_num_sn,stat_model_num_failure,stat_model_date_num_item,stat_model_date_num_failure,stat_model_count]    

if __name__=='__main__':
    print '[%s]main start...' %(time.asctime( time.localtime(time.time())))
    
    path_load = os.getenv("HOME")+'/Data/backblaze/data_bb'
    path_model = os.getenv("HOME")+'/Data/backblaze/model_file/'
    path_stat =  os.getenv("HOME")+'/Data/backblaze/stat/'
    
    data = load_file(path_load,nrows=500000)
    data_filter = filter_data(data)
    
    [m1,m2,m3,m4,m5,m6] = group_data_by_models(data_filter,path_model)
    [s1,s2,s3,s4,s5,s6] = stat_by_models(data_filter,path_stat)
    
    print '[%s]main end...' %(time.asctime( time.localtime(time.time())))
