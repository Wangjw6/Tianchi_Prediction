from data.data_loader import Dataset_Ali
from exp.exp_basic import Exp_Basic
from models.model import Model
from data.ali_dataloader import *
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, evaluate_metrics
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Predict(Exp_Basic):
    def __init__(self, args):
        super(Exp_Predict, self).__init__(args)

    def _build_model(self):

        model = Model(
            self.args.enc_in,
            self.args.dec_in,
            self.args.c_out,
            self.args.seq_len,
            self.args.pred_len,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.dropout,
            self.args.embed,
            self.args.data[:-1],
            self.args.activation,
            self.device
        )

        return model.double()

    def _get_data(self, flag=None, data_type=0, mode=0, tcfile=None):
        args = self.args
        data_type = data_type

        Data = Dataset_Ali

        df_raw = pd.DataFrame()
        if data_type == 100:
            # df_raw = pd.concat([df_raw, data_to_pd()], axis=1)
            try:
                df_raw = data_to_pd()
            except:
                df_raw = data_to_pd(path_feature='/home/project/data/SODA_train.npy',
                                    path_target='/home/project/data/SODA_label.npy')
            print('prepare real data')
        else:
            train_data_loaders = []
            test_data_loaders = []
            vali_data_loaders = []
            path_ = 'G:\\base\\aliyun\\CMIP5\\'
            path_ = '/home/project/data/CMIP5/'
            if data_type == 0:
                for mode in range(17):
                    df = data_to_pd(path_feature=path_ + str(mode) + '_train.npy',
                                    path_target=path_ + str(mode) + '_label.npy')
                    if df.isnull().values.any():
                        print('NAN!!!', mode)
                        continue
                    train_data_loader, test_data_loader, vali_data_loader = self.assist(Data,args,df[:int(df.shape[0] * 0.8)],
                                                                                        df[int(df.shape[0] * 0.8):int(
                                                                                            df.shape[0] * 0.9)],
                                                                                        df[int(df.shape[0] * 0.9):]
                                                                                        )
                    train_data_loaders.append(train_data_loader)
                    test_data_loaders.append(test_data_loader)
                    vali_data_loaders.append(vali_data_loader)
                    print('prepare CMIP5 train data', mode)
            path_ = 'G:\\base\\aliyun\\CMIP6\\'
            path_ = '/home/project/data/CMIP6/'
            if data_type == 0:
                for mode in range(15):
                    df = data_to_pd(path_feature=path_ + str(mode) + '_train.npy',
                                    path_target=path_ + str(mode) + '_label.npy')
                    if df.isnull().values.any():
                        print('NAN!!!', mode)
                        continue
                    train_data_loader, test_data_loader, vali_data_loader = self.assist(Data,args,df[:int(df.shape[0] * 0.8)],
                                                                                        df[int(df.shape[0] * 0.8):int(
                                                                                            df.shape[0] * 0.9)],
                                                                                        df[int(df.shape[0] * 0.9):]
                                                                                        )
                    train_data_loaders.append(train_data_loader)
                    test_data_loaders.append(test_data_loader)
                    vali_data_loaders.append(vali_data_loader)
                    print('prepare CMIP6 train data', mode )
            try:
                df_raw = data_to_pd()
            except:
                df_raw = data_to_pd(path_feature='/home/project/data/SODA_train.npy',
                                    path_target='/home/project/data/SODA_label.npy')
            train_data_loader, test_data_loader, vali_data_loader = self.assist(Data, args, df_raw[:int(df.shape[0] * 0.8)],
                                                                                df[int(df.shape[0] * 0.8):int(
                                                                                    df.shape[0] * 0.9)],
                                                                                df[int(df.shape[0] * 0.9):]
                                                                                )
            train_data_loaders.append(train_data_loader)
            test_data_loaders.append(test_data_loader)
            vali_data_loaders.append(vali_data_loader)


        if flag == None:
            return   train_data_loaders,  vali_data_loaders, test_data_loaders
        else:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
            data = df_data.values
            shuffle_flag = False;
            drop_last = True;
            batch_size = args.batch_size
            data_set = Data(
                flag=flag,
                data_type=100,
                tcfile=tcfile,
                data=data
            )
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )
            return data_set, data_loader

    def assist(self,Data,args,df_raw_train,df_raw_test,df_raw_vali):
        cols_data = df_raw_train.columns[1:]
        df_data = df_raw_train[cols_data]

        data = df_data.values
        train_data_set = Data(
            data_type=0,
            data=data
        )
        shuffle_flag = True;
        drop_last = True;
        batch_size = args.batch_size
        train_data_loader = DataLoader(
            train_data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        shuffle_flag = False;
        drop_last = True;
        batch_size = args.batch_size

        df_data = df_raw_test[cols_data]
        data = df_data.values
        test_data_set = Data(
            data_type=0,
            data=data
        )
        test_data_loader = DataLoader(
            test_data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        df_data = df_raw_vali[cols_data]
        data = df_data.values
        vali_data_set = Data(
            data_type=0,
            data=data
        )
        vali_data_loader = DataLoader(
            vali_data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return train_data_loader,test_data_loader,vali_data_loader
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        print('prepare data...')
        train_data_loaders, vali_data_loaders, test_data_loaders = self._get_data()
        print('Number of data loaders:', len(train_data_loaders))
        path = './checkpoints/' + setting
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            for index in range(len(train_data_loaders)):
                train_loader = train_data_loaders[index]
                train_loss = []
                begin_ = time.time()
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    iter_count += 1

                    model_optim.zero_grad()

                    batch_x = batch_x.double()  # .to(self.device)
                    batch_y = batch_y.double()

                    outputs = self.model(batch_x).view(-1, 24)
                    batch_y = batch_y[:, -self.args.pred_len:, -1].view(-1, 24)  # .to(self.device)

                    loss = criterion(outputs, batch_y)# + 0.1*corr

                    train_loss.append(loss.item())

                    loss.backward()
                    model_optim.step()
                print('INDEX Finished',index, 'train loss',np.average(train_loss),'COST',time.time()-begin_)
            train_loss = np.average(train_loss)
            vali_loss, mae, score = self.test('1')
            early_stopping(-score, self.model, path)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} score: {4:.7f}".format(
                epoch + 1, 0, np.average(train_loss), vali_loss, score))

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        print('Model is saved at', best_model_path)
        self.model.eval()
        return self.model

    def test(self, setting):
        try:
            self.test_data = self.test_data
        except:
            self.test_data, self.test_loader = self._get_data(flag='test', data_type=100)

        preds = []
        trues = []
        hiss = []
        for i, (batch_x, batch_y) in enumerate(self.test_loader):
            batch_x = batch_x.double()  # .to(self.device)
            batch_y = batch_y.double()

            outputs = self.model(batch_x).view(-1, 24).detach()
            batch_y = batch_y[:, -self.args.pred_len:, -1]  # .to(self.device)

            pred = outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y.detach().cpu().numpy()  # .squeeze()

            hiss.append(batch_x.detach().cpu().numpy())
            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, 24)
        trues = trues.reshape(-1, 24)
        data = pd.DataFrame()
        for i in range(preds.shape[0]):
            data['pred_' + str(i)] = preds[i]
            data['true_' + str(i)] = trues[i]
        # data.to_csv('result.csv')
        print('tset_typeest shape:', preds.shape, trues.shape)
        try:
            mae, mse, rmse, mape, mspe = metric(preds[:, :, -1], trues[:, :, -1])
            score = evaluate_metrics(preds[:, :, -1], trues[:, :, -1])
        except:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            score = evaluate_metrics(preds, trues)
        print('mse:{}, mae:{}, score:{}'.format(mse, mae, score))
        return mse, mae, score


    def compet(self, ):
        # print('working directory', os.getcwd())
        path = os.path.abspath(os.pardir) + '/checkpoints/' + '1'
        best_model_path = path + '/' + 'checkpoint.pth'
        print(best_model_path)

        from os import listdir
        from os.path import isfile, join
        path = os.path.abspath(os.pardir)
        print('Path', path)
        arr = os.listdir(path + '/')
        print('All the stuff')
        print(arr)

        if os.path.exists('./result/'):
            print('HAS RESULT FOLDER')
        try:
            self.model.load_state_dict(torch.load(best_model_path))
            print('Model Loaded!')
        except:
            print('!!!Can not load the mdoel')
        self.model.eval()

        test_path = './tcdata/enso_round1_test_20210201/'

        files = os.listdir(test_path)
        test_feas_dict = {}
        for file in files:
            test_feas_dict[file] = np.load(test_path + file)

        test_predicts_dict = {}
        for file_name, val in test_feas_dict.items():
            batch_x = torch.tensor(val, dtype=torch.double).permute(0, 3, 1, 2).reshape(1, 12, -1)
            outputs = self.model(batch_x)
            pred = outputs.detach().cpu().numpy()
            pred = pred.reshape(-1, )
            test_predicts_dict[file_name] = pred

        for file_name, val in test_predicts_dict.items():
            np.save('./result/' + file_name, val)
        print('Prediction done! See below:')
        arr = os.listdir('./result/')
        print(arr)
        return


