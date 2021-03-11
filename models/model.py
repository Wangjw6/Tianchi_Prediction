import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tcn import TemporalConvNet
from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention,AttentionLayer
from models.embed import DataEmbedding

class Model(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, out_len,d_model=120, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.0,  embed='fixed', data='Ali_00', activation='gelu',
                device=torch.device('cuda:0')):
        super(Model, self).__init__()
        self.pred_len = out_len
        self.tcn = TemporalConvNet(d_model, [d_model,d_model,d_model], kernel_size=2, dropout=dropout)
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        # Attention
        Attn =  FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, attention_dropout=dropout),
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
    def forward(self, x_enc):
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(enc_out)
        dec_out = self.predict(enc_out.view(-1,self.hidden))
        return dec_out
