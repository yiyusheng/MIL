# merge data into one file for years
import pandas as pd
import os

# %%load_data
def load_file(path):
    data = pd.read_csv(path,sep=',')
    print 'Read %s success...' %(path)
    return data

# %% read files
def merge_dir(dir_name):
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
    data.to_csv(path+dir_name,index=0)
    print "Merge %s success..." %(dir_name)
    return data
    
# %%
if __name__=='__main__':
    
    data13 = merge_dir('data_2013')
    data15 = merge_dir('data_2015')
    data16 = merge_dir('data_2016')
    data17 = merge_dir('data_2017')
    data18 = merge_dir('data_2018')
    data19 = merge_dir('data_2019')
    
    data_list = [data13,data15,data16,data17,data18,data19]
    data_bb = pd.concat(data_list,sort=False)
    data_bb.to_csv('/home/yiyusheng/Data/backblaze/data_bb',index=0)