import argparse
import os
from exp.exp_informer import Exp_Informer
from data.generate_data import *
parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str,  default='informer',help='model of the experiment')

parser.add_argument('--data', type=str,  default='Ali_00', help='data') #Ali_00: true data
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='location of the data file')    
parser.add_argument('--features', type=str, default='M', help='features [S, M]')
parser.add_argument('--target', type=str, default='OT', help='target feature')

parser.add_argument('--seq_len', type=int, default=12, help='input series length')
parser.add_argument('--label_len', type=int, default=12, help='help series length')
parser.add_argument('--pred_len', type=int, default=24, help='predict series length')
parser.add_argument('--enc_in', type=int, default=4*24*72, help='encoder input size')
parser.add_argument('--dec_in', type=int, default= 1, help='decoder input size')
parser.add_argument('--c_out', type=int, default= 1, help='output size')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=120, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=1, help='prob sparse factor')

parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention [prob, full]')
parser.add_argument('--embed', type=str, default='fixed', help='embedding type [fixed, learned]')
parser.add_argument('--activation', type=str, default='relu',help='activation')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

parser.add_argument('--itr', type=int, default=2, help='each params run iteration')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='input data batch size')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')

parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()


Exp = Exp_Informer

for ii in range(args.itr):
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_eb{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.embed, args.des, ii)
    
    # try:
    #     nc_npy()
    # except:
    #     print('no npy prcoessed')

    exp = Exp(args)
    print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train('1')

    print('>>>>>>>testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test('1')


    exp.compet()
