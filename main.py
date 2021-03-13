import argparse
import os
from exp.exp_predict import Exp_Predict
from data.generate_data import *
from random import seed
import torch
import zipfile
parser = argparse.ArgumentParser(description='TianChi AI')

parser.add_argument('--model', type=str,  default='tcp',help='model of the experiment')

parser.add_argument('--data', type=str,  default='Ali_00', help='data') #Ali_00: true data
parser.add_argument('--root_path', type=str, default='None', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='None', help='location of the data file')
parser.add_argument('--target', type=str, default='nino', help='target feature')

parser.add_argument('--seq_len', type=int, default=12, help='input series length')
parser.add_argument('--label_len', type=int, default=12, help='help series length')
parser.add_argument('--pred_len', type=int, default=24, help='predict series length')
parser.add_argument('--enc_in', type=int, default=4*24*72, help='encoder input size')
parser.add_argument('--dec_in', type=int, default= 1, help='decoder input size')
parser.add_argument('--c_out', type=int, default= 1, help='output size')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=120, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--embed', type=str, default='fixed', help='embedding type [fixed, learned]')
parser.add_argument('--activation', type=str, default='relu',help='activation')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')

parser.add_argument('--itr', type=int, default=2, help='each params run iteration')
parser.add_argument('--train_epochs', type=int, default=3, help='train epochs')
parser.add_argument('--batch_size', type=int, default=16, help='input data batch size')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.00004, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')

parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')

args = parser.parse_args()
def compress(res_dir='./result', output_dir='result.zip'):
    import zipfile
    z = zipfile.ZipFile(output_dir, 'w')
    for d in os.listdir(res_dir):
        z.write(res_dir + os.sep + d)
    z.close()

Exp = Exp_Predict

if __name__ == '__main__':
    seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    exp = Exp(args)
    print('>>>>>>>start training>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train('1')

    print('>>>>>>>testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test('1')
    # try:
    #     exp.compet()
    #
    #     compress()
    #     print('Zip done')
    #
    #     arr = os.listdir('./')
    #     print(arr)
    # except:
    #     print('NO GAME HERE')