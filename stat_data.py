#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:08:19 2019

@author: yiyusheng

Stat data
"""

import os
import pandas as pd

path = os.getenv("HOME")+'/Data/backblaze/model_file/'
model_name = ['ST4000DM000', 'HGST HMS5C4040BLE640','ST12000NM0007',
              'HGST HMS5C4040ALE640','ST8000NM0055','ST8000DM002']
mn = model_name[3]

path = os.getenv("HOME")+'/Data/backblaze/'
mn = 'data_bb'
data = pd.read_csv(path+mn,sep=',')

