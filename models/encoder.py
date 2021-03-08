import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, c_in,kernel_size=3):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=kernel_size,
                                  padding=padding)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu#F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask = attn_mask
        ))

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y)

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None,tcn_layers=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer
        self.tcn_layers = tcn_layers
        self.lstm_layers = nn.LSTM(input_size=64,hidden_size=64,batch_first=True)
    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        if self.tcn_layers is not None:
            # x, (hn, cn) = self.lstm_layers(x )
            x = self.tcn_layers(x)
            # x = self.attn_layers[-1](x)
        else:
            if self.conv_layers is not None:
                for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                    # x = attn_layer(x, attn_mask=attn_mask)
                    x = conv_layer(x)
                x = self.attn_layers[-1](x)
            else:
                for attn_layer in self.attn_layers:
                    x = attn_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x