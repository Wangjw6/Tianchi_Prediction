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

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', data='ETTh'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if data=='ETTm':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()

        return self.month_embed(x[:,:,0])
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

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
    def __init__(self, c_in, d_model, embed_type='fixed', data='ETTh', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.spatial_embedding = SpatialEmbedding(c_in=4,c_out=2,d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        x = self.spatial_embedding(x)+self.position_embedding(x)
        return x
