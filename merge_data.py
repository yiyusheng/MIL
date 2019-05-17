#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:17:42 2019

@author: yiyusheng

merge data into one file for years
"""

import pandas as pd
import os,time

# %%load_data
def load_file(path,nrows=-1):
    if nrows==-1:
        data = pd.read_csv(path,sep=',')
        print '[%s]Read %s success...' %(time.asctime(time.localtime(time.time())),path)
    else:
        data = pd.read_csv(path,sep=',',nrows=nrows)
        print '[%s]Read %s %d lines success...' %(time.asctime(time.localtime(time.time())),path,nrows)       
    return data

def get_colnames(data):
    cn = data.columns.values.tolist()  
    name_meta = [i for i in cn if 'smart' not in i]
    name_smart = [i for i in cn if 'smart' in i]
    return [name_meta,name_smart]

# %% read files
def merge_dir(dir_name,save=1):
    path = '/home/yiyusheng/Data/backblaze/'
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

# %%
if __name__=='__main__':
    print '[%s]main start...' %(time.asctime( time.localtime(time.time())))
    
    data13 = merge_dir('data_2013')
    data15 = merge_dir('data_2015')
    data16 = merge_dir('data_2016')
    data17 = merge_dir('data_2017')
    data18 = merge_dir('data_2018')
    data19 = merge_dir('data_2019')
#    data_list = [data13,data15,data16,data17,data18,data19]
#    data_all = merge_all(data_list)

    print '[%s]main end...' %(time.asctime( time.localtime(time.time())))
