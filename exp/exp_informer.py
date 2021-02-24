from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute,Dataset_Ali,Dataset_Alitest
from exp.exp_basic import Exp_Basic
from models.model import Informer
from data.ali_dataloader import *
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric,evaluate_metrics
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
        }
        if self.args.model=='informer':
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.data[:-1],
                self.args.activation,
                self.device
            )
        
        return model.double()

    def _get_data(self, flag,data_type=0,mode=0,tcfile=None):
        args = self.args
        data_type = data_type
        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'Ali_00':Dataset_Ali,
        }
        if mode==0:
            Data = data_dict[self.args.data]
        else:
            Data = Dataset_Alitest
        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_type=data_type,
            tcfile=tcfile
        )
      
        print(flag,len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.double()#.to(self.device)
            batch_y = batch_y.double()
            
            batch_x_mark = batch_x_mark.double()#.to(self.device)
            batch_y_mark = batch_y_mark.double()#.to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double()#.to(self.device)
            # encoder - decoder
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            batch_y = batch_y[:,-self.args.pred_len:,:]#.to(self.device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            try:
                loss = criterion(pred[:,:,-1], true[:,:,-1])
            except:
                loss = criterion(pred, true[:, :, -1])

            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
        
    def train(self, setting):
        print('prepare train data...')
        train_data, train_loader = self._get_data(flag = 'train')
        print('prepare validate data...')
        vali_data, vali_loader = self._get_data(flag = 'val')
        print('prepare test data...')
        test_data, test_loader = self._get_data(flag = 'test')

        path = './checkpoints/'+setting
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                
                batch_x = batch_x.double()#.to(self.device)
                batch_y = batch_y.double()

                batch_x_mark = batch_x_mark.double()#.to(self.device)
                batch_y_mark = batch_y_mark.double()#.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:-1]).double()
                dec_inp = torch.cat([batch_y[:,:self.args.label_len,:-1], dec_inp], dim=1).double()#.to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark).view(-1,24)

                batch_y = batch_y[:,-self.args.pred_len:,-1].view(-1,24)#.to(self.device)
                loss = criterion(outputs, batch_y)
                # loss2 = criterion(outputs[:,:,-1], batch_y[:,:,-1])
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test',data_type=100)
        
        self.model.eval()
        
        preds = []
        trues = []
        hiss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.double()#.to(self.device)
            batch_y = batch_y.double()
            batch_x_mark = batch_x_mark.double()#.to(self.device)
            batch_y_mark = batch_y_mark.double()#.to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:,-self.args.pred_len:,:]).double()
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).double()#.to(self.device)
            # encoder - decoder
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark).view(-1,24)
            batch_y = batch_y[:,-self.args.pred_len:,-1]#.to(self.device)
            
            pred = outputs.detach().cpu().numpy()#.squeeze()
            true = batch_y.detach().cpu().numpy()#.squeeze()

            hiss.append(batch_x.detach().cpu().numpy())
            preds.append(pred)
            trues.append(true)

        preds = np.array(preds)
        hiss = np.array(hiss)
        trues = np.array(trues)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, 24)
        trues = trues.reshape(-1, 24)

        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        try:
            mae, mse, rmse, mape, mspe = metric(preds[:,:,-1], trues[:,:,-1])
            score = evaluate_metrics(preds[:,:,-1], trues[:,:,-1])
        except:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            score = evaluate_metrics(preds, trues)
        print('mse:{}, mae:{}, score:{}'.format(mse, mae,score))
        return


        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)

        return

    def compet(self, ):
        # print('working directory', os.getcwd())
        path = os.path.abspath(os.pardir) + '/checkpoints/' + '1'
        best_model_path = path + '/' + 'checkpoint.pth'
        print(best_model_path)

        # from os import listdir
        # from os.path import isfile, join
        # onlyfiles = [f for f in listdir(path + '/') if isfile(join(path + '/', f))]
        # print('Files')
        # print(onlyfiles)

        try:
            self.model.load_state_dict(torch.load(best_model_path))
        except:
            print('!!!Can not load the mdoel')
        self.model.eval()
        for d in os.listdir('../tcdata/enso_round1_test_20210201'):
            print('Predicting: ', d)
            test_data, test_loader = self._get_data(flag='test', mode=1,
                                                    tcfile='../tcdata/enso_round1_test_20210201/' + d)

            preds = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                print('Round', i)
                batch_x = batch_x.double()  # .to(self.device)
                batch_y = batch_y.double()
                batch_x_mark = batch_x_mark.double()  # .to(self.device)
                batch_y_mark = batch_y_mark.double()  # .to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).double()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).double()  # .to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                pred = outputs.detach().cpu().numpy()
                print('Pred shape', pred.shape)
                preds.append(pred)

            preds = np.array(preds)
            print('test shape:', preds.shape)
            preds = preds.reshape(-1, 24)
            print('test shape:', preds.shape)

            # result save
            folder_path = 'result/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path + '/' + d, preds)

        def compress(res_dir='./result', output_dir='result.zip'):
            z = zipfile.ZipFile(output_dir, 'w')
            for d in os.listdir(res_dir):
                z.write(res_dir + os.sep + d)
            z.close()

        compress()
        print('done!')
        return