#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:17:42 2019

@author: yiyusheng

merge data into one file for years
"""

import pandas as pd
import os,time,sys
from base_func import load_file,get_colnames

# %% read files
def merge_dir(dir_name,save=1):
    path = os.getenv('HOME')+'/Data/backblaze/'
    dir_path = path+'date_file/'+dir_name+'/'
    fname = os.listdir(dir_path)
    fname.sort()
    data_list = list()
    for f in fname:
        d = load_file(dir_path+f)
        data_list.append(d)
    data = pd.concat(data_list,sort=False)
    data = data.sort_values(['date','model','serial_number'])
    if save==1:
        data.to_csv(path+'year_file/'+dir_name,index=0)
    print "Merge %s success..." %(dir_name)
    return data
    
def merge_all(data_list,save=1):
    data = pd.concat(data_list,sort=False)
    if save==1:
        data.to_csv('/home/yiyusheng/Data/backblaze/data_all',index=0)
    return data

# %% group data by models
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
    print '[%s]%s done...' %(time.asctime(time.localtime(time.time())),sys._getframe().f_code.co_name)
    return data_models

def stat_by_models(data_filter,path_stat,save=1):  
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


# %%
if __name__=='__main__':
    print '[%s]main start...' %(time.asctime( time.localtime(time.time())))
    
    data13 = merge_dir('data_2013')
    data15 = merge_dir('data_2015')
    data16 = merge_dir('data_2016')
    data17 = merge_dir('data_2017')
    data18 = merge_dir('data_2018')
    data19 = merge_dir('data_2019')
    data_list = [data13,data15,data16,data17,data18,data19]
    data_all = merge_all(data_list)
    
    path_model = os.getenv("HOME")+'/Data/backblaze/model_file/'
    path_stat =  os.getenv("HOME")+'/Data/backblaze/stat/'    

    [m1,m2,m3,m4,m5,m6] = group_data_by_models(data_all,path_model)
    [s1,s2,s3,s4,s5,s6] = stat_by_models(data_all,path_stat)
    print '[%s]main end...' %(time.asctime( time.localtime(time.time())))
