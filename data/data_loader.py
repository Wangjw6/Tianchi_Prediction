import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from data.ali_dataloader import *
import warnings
warnings.filterwarnings('ignore')


class Dataset_Ali(Dataset):
    def __init__(self,data,  flag=None,  data_type=1,tcfile=None):
        # size [seq_len, label_len pred_len]
        # info

        self.seq_len = 12
        self.label_len = 12
        self.pred_len = 24
        self.data_type = data_type
        self.data = data
        self.__read_data__()


    def __read_data__(self, ):
        self.data_x =  self.data[:,:-1]
        self.data_y = self.data



    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]


        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

