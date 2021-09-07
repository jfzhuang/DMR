import sys
import torch
from torch import nn
from torch.nn import functional as F

from mmseg.models.transformer.pos_embedding import positionalencoding3d
from mmseg.models.transformer.detr import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerHEDecoder,
    TransformerHELayer,
)


class STF(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.depth = depth

        enc_layer = TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim)
        self.encoder = TransformerEncoder(enc_layer, num_layers=depth)
        dec_layer = TransformerDecoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim)
        self.decoder = TransformerDecoder(dec_layer, num_layers=depth)
        self.query = nn.Parameter(torch.rand([1, 512, 64, 128]), requires_grad=True)

    def get_pe(self, feat, n, c, t, h, w):
        pe = positionalencoding3d(c, t, h, w)
        pe = pe.to(feat.device).unsqueeze(0)
        if n > 1:
            pe = pe.repeat([n, 1, 1, 1, 1])
        return pe.detach()

    def forward(self, x_enc):
        x_enc_0, x_enc_1, x_enc_2 = x_enc
        x_enc_0 = x_enc_0.unsqueeze(2)
        x_enc_1 = x_enc_1.unsqueeze(2)
        x_enc_2 = x_enc_2.unsqueeze(2)

        x_enc = torch.cat([x_enc_0, x_enc_1, x_enc_2], dim=2)
        n, c, t_e, h, w = x_enc.shape

        x_dec = x_enc_1
        t_q = x_dec.shape[2]

        pe_dec = self.query
        if pe_dec.shape[2] != h or pe_dec.shape[3] != w:
            pe_dec = F.interpolate(pe_dec, size=(h, w), mode='bilinear')
        if n > 1:
            pe_dec = pe_dec.repeat(n, 1, 1, 1)
        pe_dec = pe_dec.unsqueeze(2)

        pe_enc = self.get_pe(x_enc, n, c, t_e, h, w)

        x_enc = self.encoder(x_enc, pos=pe_enc)
        x_dec = self.decoder(x_dec, x_enc, pos=pe_enc, query_pos=pe_dec)
        x_dec = x_dec.squeeze(2)

        return x_dec


if __name__ == '__main__':
    import random
    import numpy as np

    model = STF(dim=512, depth=2, heads=2, mlp_dim=512)
    model.cuda().eval()

    x_enc = []
    for i in range(3):
        x_enc.append(torch.rand(1, 512, 128, 256).cuda())

    with torch.no_grad():
        x_dec = model(x_enc)
        print(x_dec.shape)
