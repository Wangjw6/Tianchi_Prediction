import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from data.ali_dataloader import *
import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_type=0,
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True):
        # size [seq_len, label_len pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            data = scaler.fit_transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
        df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
        data_stamp = df_stamp.drop(['date'],1).values
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_type=0,
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True):
        # size [seq_len, label_len pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            data = scaler.fit_transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row:row.month,1)
        df_stamp['day'] = df_stamp.date.apply(lambda row:row.day,1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row:row.weekday(),1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row:row.hour,1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row:row.minute,1)
        df_stamp['minute'] = df_stamp.minute.map(lambda x:x//15)
        data_stamp = df_stamp.drop(['date'],1).values
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1


class Dataset_Ali(Dataset):
    def __init__(self,  flag='train', size=None,root_path=None,data_path=None,
                 features='S', data_type=1,tcfile=None,
                 target='OT', scale=True):
        # size [seq_len, label_len pred_len]
        # info

        self.seq_len = 12
        self.label_len = 12
        self.pred_len = 24
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_type
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale

        self.__read_data__()

    def __read_data__(self):
        scaler = StandardScaler()
        df_raw = pd.DataFrame()
        if self.data_type==90 or self.data_type==100:
            # df_raw = pd.concat([df_raw, data_to_pd()], axis=1)
            try:
                df_raw = data_to_pd()
            except:
                df_raw = data_to_pd(path_feature='/home/project/data/SODA_train.npy',path_target='/home/project/data/SODA_label.npy')
            print('prepare real data')
        path_ =  'G:\\base\\aliyun\\CMIP5\\'
        path_ = '/home/project/data/CMIP5/'
        if self.data_type ==  0:
            for mode in range(17):
                df = data_to_pd(path_feature =path_+str(mode)+'_train.npy',path_target =path_ +str(mode)+'_label.npy')
                if df.isnull().values.any():
                    print('NAN!!!',mode)
                    continue
                df_raw = pd.concat([df_raw, df], axis=0)
                print('prepare CMIP5 data', mode,df_raw.shape[0])

                # df_raw = data_to_pd(path_feature =path_feature+str(mode)+'_train.npy',path_target =path_target+str(mode)+'_label.npy')
        if self.data_type == 0:
            path_ = 'G:\\base\\aliyun\\CMIP6\\'
            path_ = '/home/project/data/CMIP6/'
            for mode in range(14 ):
                df = data_to_pd(path_feature=path_ + str(mode) + '_train.npy',
                                path_target=path_ + str(mode) + '_label.npy')
                if df.isnull().values.any():
                    print('NAN!!!',mode)
                    continue
                df_raw = pd.concat([df_raw, df], axis=0)

                print('prepare CMIP6 data', mode, df_raw.shape[0])


        border1s = [0, int(df_raw.shape[0]*0.8) - self.seq_len, int(df_raw.shape[0]*0.9)  - self.seq_len]
        border2s = [int(df_raw.shape[0]*0.8), int(df_raw.shape[0]*0.9)  , int(df_raw.shape[0])]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        data = df_data.values

        datax = data[:,:-1]
        self.data_x = datax[border1:border2]
        self.data_y = data[border1:border2]
        if self.data_type==90 or self.data_type==100:
            self.data_x = datax
            self.data_y = data


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = 0#self.data_stamp[s_begin:s_end]
        seq_y_mark = 0#self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class Dataset_Alitest(Dataset):
    def __init__(self, flag='train', size=None, root_path=None, data_path=None,
                 features='S', data_type=1,tcfile=None,
                 target='OT', scale=True):
        # size [seq_len, label_len pred_len]
        # info

        self.seq_len = 12
        self.label_len = 12
        self.pred_len = 24
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = 1
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale

        self.__read_data__(tcfile=tcfile)


    def __read_data__(self,tcfile=None):
        pd_data = pd.DataFrame()
        t = []
        y = 1800
        for m in range(12):
            if (m + 1) % 12 == 0:
                t.append('{}-{}-{}'.format(y, 12, '01'))
                y += 1
            else:
                if m + 1 >= 10:
                    t.append('{}-{}-{}'.format(y, (m + 1) % 12, '01'))
                else:
                    t.append('{}-0{}-{}'.format(y, (m + 1) % 12, '01'))

        pd_data['date'] = pd.to_datetime(t, errors='coerce')
        print('Load data...')
        feature = np.load(tcfile)
        print('Load data shape',feature.shape)
        f = ['sst', 't300', 'ua', 'va']
        for i in range(feature.shape[-1]):
            # spatial-wise feature accumulation
            ## mean
            s = (feature[:, :, :, i])
            s1 = s.reshape(12, -1)
            for j in range(s1.shape[1]):
                pd_data[f[i] + '_' + str(j)] = s1[:, j].tolist()

        pd_data['OT'] = [0 for _ in range(12)]

        cols_data = pd_data.columns[1:]
        df_data = pd_data[cols_data]

        self.data_x = df_data.values[:,:-1]
        self.data_y = df_data.values

        print('read data shape',self.data_x.shape)



    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = 0#self.data_stamp[s_begin:s_end]
        seq_y_mark = 0#self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


    def __len__(self):
        return len(self.data_x)# - self.seq_len - self.pred_len + 1