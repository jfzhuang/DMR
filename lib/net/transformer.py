import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

from lib.net.pos_embedding import positionalencoding3d
from lib.net.transformer_block import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerHELayer
)


class CRM(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, pe_dim, block_size=[16, 16]):
        super().__init__()
        enc_layer = TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, block_size=block_size)
        self.encoder = TransformerEncoder(enc_layer, num_layers=depth)
        dec_layer = TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, block_size=block_size)
        self.decoder = TransformerDecoder(dec_layer, num_layers=depth)
        
        self.pe_enc = nn.Parameter(torch.rand([1, dim, 3, pe_dim[0], pe_dim[1]]), requires_grad=True)
        self.pe_dec = nn.Parameter(torch.rand([1, dim, 1, pe_dim[0], pe_dim[1]]), requires_grad=True)
            
    def forward(self, x_enc_0, x_enc_1, x_enc_2):
        x_enc = torch.cat([x_enc_0, x_enc_1, x_enc_2], dim=2)
        n, c, t_e, h, w = x_enc.shape
        
        pe_enc = self.pe_enc
        pe_dec = self.pe_dec
        
        if pe_enc.shape[2] != h or pe_enc.shape[3] != w:
            pe_enc = F.interpolate(pe_enc, size=(3, h, w), mode='trilinear')
            pe_dec = F.interpolate(pe_dec, size=(1, h, w), mode='trilinear')
                    
        if n > 1:
            pe_enc = pe_enc.repeat(n, 1, 1, 1, 1)
            pe_dec = pe_dec.repeat(n, 1, 1, 1, 1)        
        
        x_enc = self.encoder(x_enc, pos=pe_enc)
        x_dec = self.decoder(x_enc_1, x_enc, pos=pe_enc, query_pos=pe_dec)
        x_dec = x_dec.squeeze(2)

        return x_dec

class ERM(nn.Module):
    def __init__(self, dim, mlp_dim, KL, KH, num_class):
        super().__init__()
        self.HEdecoder = TransformerHELayer(d_model=dim, dim_feedforward=dim//2)
        self.memory_key = nn.Parameter(torch.rand([1, KL*num_class, dim]), requires_grad=True)
        self.memory_value = nn.Parameter(torch.rand([1, KH*num_class, dim]), requires_grad=True)
        
    def forward(self, x):
        n, c, h, w = x.shape
        
        memory_key = self.memory_key.repeat([n, 1, 1])
        memory_value = self.memory_value.repeat([n, 1, 1])
        
        x = self.HEdecoder(query=x, key=memory_key, value=memory_value)

        return x

if __name__ == '__main__':
    import random
    import numpy as np

    model = CRM(dim=512, depth=1, heads=2, mlp_dim=256, pe_dim=(128, 256), use_checkpoint=True)
    model.cuda().eval()

    x_enc = torch.rand(1, 512, 1, 128, 256).cuda()
#     with torch.no_grad():
    while True:
        preds = model(x_enc, x_enc, x_enc)
        print(preds.shape)
    
#     model = ERM(dim=512, mlp_dim=256, KL=10, KH=10, num_class=19)
#     model.cuda().eval()

#     x = torch.rand(1, 512, 128, 256).cuda()
#     with torch.no_grad():
#         preds = model(x)
#         print(preds.shape)