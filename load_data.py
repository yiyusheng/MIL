#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:22:22 2019

@author: yiyusheng

load data
"""

import pandas as pd
import os,time

def load_file(path):
    data = pd.read_csv(path,sep=',')
    print '[%s]Read %s success...' %(time.asctime( time.localtime(time.time())),path)
    return data

if __name__=='__main__':
    print '[%s]main start...' %(time.asctime( time.localtime(time.time())))
    path = os.getenv("HOME")+'/Data/backblaze/year_file/'
    data15 = load_file(path+'data_2015')
#    data16 = load_file(path+'data_2016')
#    data17 = load_file(path+'data_2017')
#    data18 = load_file(path+'data_2018')
#    data19 = load_file(path+'data_2019')
#    data_list = [data15,data16,data17,data18,data19]
#    data_bb = pd.concat(data_list,sort=False)
