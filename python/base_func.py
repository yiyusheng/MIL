#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:17:42 2019

@author: yiyusheng

merge data into one file for years
"""

import pandas as pd
import time

# %%load_data
def load_file(path,nrows=-1,quiet=True):
    if nrows==-1:
        data = pd.read_csv(path,sep=',')
        if quiet==False:
            print('[%s]Read %s success...' %(time.asctime(time.localtime(time.time())),path))
    else:
        data = pd.read_csv(path,sep=',',nrows=nrows)
        if quiet==False:
            print('[%s]Read %s %d lines success...' %(time.asctime(time.localtime(time.time())),path,nrows))       
    return data

def get_colnames(data):
    cn = data.columns.values.tolist()  
    name_meta = [i for i in cn if 'smart' not in i]
    name_smart = [i for i in cn if 'smart' in i]
    return [name_meta,name_smart]


# %%
if __name__=='__main__':
    pass
