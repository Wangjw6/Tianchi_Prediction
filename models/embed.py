import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SpatialEmbedding(nn.Module):
    def __init__(self, c_in, c_out=1,d_model=120):
        super(SpatialEmbedding, self).__init__()
        self.c_out = c_out
        padding = 1 if torch.__version__>='1.5.0' else 2

        conv_layes = []
        while c_out>=1:
            if c_out==self.c_out:
                c_in = 4
            else:
                c_in = c_out+1
            conv_layes.append(nn.Conv2d(in_channels=c_in, out_channels=c_out,kernel_size =(3, 3), stride=(1, 1), padding=(1, 1)))
            c_out-=1
        self.spatialConv = nn.ModuleList(conv_layes)

        self.spatialPool = nn.MaxPool2d((2, 2), stride=(1, 1))
        self.fc = nn.Linear(22*70,d_model)
        self.d_model = d_model
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        out = []
        for j in range(x.size(1)):
            conv = x[:,j,:].view(-1,4,24,72)
            conv[:,0,:,:] = conv[:,0,:,:]/2.
            conv[:, 1, :, :] = conv[:,1,:,:]/50.
            conv[:,2,:,:] = conv[:,2,:,:]/200.
            conv[:, 3, :, :] = conv[:, 3, :, :] / 200.
            for i in range(len(self.spatialConv)):
                conv = self.spatialConv[i](conv)
                conv = self.spatialPool(conv)

            conv = self.fc(conv.view(-1,22*70))
            out.append(conv)
        out = torch.cat(out,1)
        # print(out[:,2].reshape(-1,)-out.view(-1,x.size(1),self.d_model)[:,0,2].reshape(-1,))
        out = out.view(-1,x.size(1),self.d_model)
        return out
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.spatial_embedding = SpatialEmbedding(c_in=4,c_out=2,d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.spatial_embedding(x)+self.position_embedding(x)
        return x
