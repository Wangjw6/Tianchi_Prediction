import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tcn import TemporalConvNet
from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=120, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.0, attn='prob', embed='fixed', data='ETTh', activation='gelu', 
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.tcn = TemporalConvNet(d_model, [d_model,d_model,d_model], kernel_size=2, dropout=dropout)
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, data, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, data, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout), 
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ],
            tcn_layers=self.tcn,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.hidden  = d_model*12
        self.predict = nn.Linear(self.hidden, 24, bias=True)
        self.d_model = d_model
        self.activation = F.relu
        self.predicts = []
        for k in range(4):
            self.predicts.append(nn.Linear(self.hidden, (k+1)*6, bias=True))
        self.predicts = nn.ModuleList(self.predicts)
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print('enc_out',enc_out.shape)
        # dec_out = self.predict(enc_out.view(-1,self.hidden))
        # print('dec_out', dec_out.shape)
        # dec_out = dec_out + self.activation(dec_out)
        dec_out = []
        for k in range(len(self.predicts)):
            dec_out.append(self.predicts[k](enc_out.view(-1,self.hidden)))
        return dec_out#[:,-self.pred_len:,:] # [B, L, D]
