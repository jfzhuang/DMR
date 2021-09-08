import torch
from torch import nn
import torch.nn.functional as F

from lib.net.transformer import STF, Transformer

from mmseg.models import build_segmentor
from mmcv.utils import Config


class SegRefine(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.num_classes = num_classes

        config_path = '/code/mmsegmentation/configs/pspnet_r50-d8.py'
        cfg = Config.fromfile(config_path)
        self.pspnet = build_segmentor(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

        self.STF = STF(dim=512, depth=1, heads=2, mlp_dim=256, pe_dim=(128, 256), use_checkpoint=True)
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.weight_init()

    def weight_init(self):
        weight_path = '/data/jfzhuang/VSPW_480p/work_dirs_v2/psp50_v2/psp50_trained.pth'
        weight = torch.load(weight_path, map_location='cpu')
        new_weight = {}
        for k, v in weight.items():
            k = k.replace('pspnet.', '')
            new_weight[k] = v
        del new_weight['decode_head.conv_seg.weight']
        del new_weight['decode_head.conv_seg.bias']
        self.pspnet.load_state_dict(new_weight, False)

        for p in self.STF.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, im_list, gt=None):
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

        feat_query = self.STF(feat_0, feat_1, feat_2)
        feat_query = self.pspnet.decode_head.cls_seg(feat_query)
        feat_query = F.interpolate(feat_query, scale_factor=8, mode='bilinear')
        
        if gt is not None:
            loss = self.semantic_loss(feat_query, gt)
            loss = loss.unsqueeze(0)
            return loss

        return feat_query
    
    def forward_return_feat(self, im_list):
        im_feat = []
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

        feat_query = self.STF(feat_0, feat_1, feat_2)
        logits = self.pspnet.decode_head.cls_seg(feat_query)

        return feat_query, logits


if __name__ == '__main__':
    model = SegRefine()
    model.cuda().eval()

    im_list = []
    for i in range(3):
        im = torch.zeros([1, 3, 1024, 2048]).cuda()
        im_list.append(im)

#     with torch.no_grad():
    while True:
        out = model(im_list)
        print(out.shape)
