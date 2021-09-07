import os 
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from lib.net.transformer import STF, MAR

from mmseg.models import build_segmentor
from mmcv.utils import Config


class SegRefine(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.num_classes = num_classes

        config_path = '/code/mmsegmentation/configs/pspnet_r50-d8.py'
        cfg = Config.fromfile(config_path)
        self.pspnet = build_segmentor(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

        self.STF = STF(dim=512, depth=1, heads=2, mlp_dim=256, pe_dim=(128, 256))
        self.MAR = MAR(dim=512, mlp_dim=256, KL=10, KH=10, num_class=19)
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=255)

        self.weight_init()

    def weight_init(self):
        weight_path = '/data/jfzhuang/VSPW_480p/work_dirs_v2/ST_Memory/exp_new/cityscapes/psp50/STF/best.pth'
        weight = torch.load(weight_path, map_location='cpu')
        pspnet_weight = {}
        STF_weight = {}
        for k, v in weight.items():
            if 'pspnet' in k:
                k = k.replace('pspnet.', '')
                pspnet_weight[k] = v
            elif 'STF' in k:
                k = k.replace('STF.', '')
                STF_weight[k] = v            
        del pspnet_weight['decode_head.conv_seg.weight']
        del pspnet_weight['decode_head.conv_seg.bias']
        self.pspnet.load_state_dict(pspnet_weight, False)
        self.STF.load_state_dict(STF_weight, False)

        for p in self.MAR.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        memory_path = '/code/ST_Memory/exp_new/cityscapes/psp50/generate_memory/output'

        key = np.load(os.path.join(memory_path, 'key.npy'))
        key = torch.from_numpy(key)
        self.MAR.memory_key.data = key.unsqueeze(0)

        value_tmp = np.load(os.path.join(memory_path, 'value.npy'))
        value_tmp = torch.from_numpy(value_tmp)
        value = []
        for i in range(19):
            value.append(value_tmp[i:i+1, :].repeat(10, 1))
        value = torch.cat(value, dim=0)
        self.MAR.memory_value.data = value.unsqueeze(0)

    def forward(self, im_list):
        with torch.no_grad():
            feat_0 = self.pspnet.backbone(im_list[0])
            feat_0 = self.pspnet.decode_head(feat_0, return_feat=True)
            feat_0 = feat_0.unsqueeze(2)
            feat_1 = self.pspnet.backbone(im_list[1])
            feat_1 = self.pspnet.decode_head(feat_1, return_feat=True)
            feat_1 = feat_1.unsqueeze(2)
            feat_2 = self.pspnet.backbone(im_list[2])
            feat_2 = self.pspnet.decode_head(feat_2, return_feat=True)
            feat_2 = feat_2.unsqueeze(2)
            feat_1 = self.STF(feat_0, feat_1, feat_2)
            
        feat_1 = self.MAR(feat_1)
        pred = self.pspnet.decode_head.cls_seg(feat_1)
        pred = F.interpolate(pred, scale_factor=8, mode='bilinear')

        return pred


if __name__ == '__main__':
    import numpy as np

    model = SegRefine()
    model.cuda().eval()

    im = torch.zeros([1, 3, 1024, 2048]).cuda()

    while True:
        out = model([im, im, im])
        print(out.shape)

    # im_list = []
    # for i in range(3):
    #     im = torch.zeros([1, 3, 1024, 2048]).cuda()
    #     im_list.append(im)

    # while True:
    #     with torch.no_grad():
    #         out = model(im_list)
    #         print(out.shape)
