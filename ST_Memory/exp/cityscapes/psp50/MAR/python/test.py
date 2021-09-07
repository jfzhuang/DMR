import os
import sys
import cv2
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.metrics import runningScore
from lib.datasets.CityscapesDataset import CityscapesDataset
from exp_new.cityscapes.psp50.MAR.model.refine import SegRefine

from mmseg.models import build_segmentor
from mmcv.utils import Config


def get_arguments():
    parser = argparse.ArgumentParser(description="Test the DAVSS")
    ###### general setting ######
    parser.add_argument("--data_path", type=str, help="path to the data")
    parser.add_argument("--im_path", type=str, help="path to the images")
    parser.add_argument("--model_weight", type=str, help="path to the trained model")

    ###### inference setting ######
    parser.add_argument("--num_workers", type=int, help="num of cpus used")

    return parser.parse_args()


def test():
    args = get_arguments()
    print(args)
    
    net = SegRefine()
    
    weight = torch.load(args.model_weight, map_location='cpu')
    net.load_state_dict(weight, True)
    net.cuda().eval()

    test_data = CityscapesDataset(args.data_path, args.im_path, 'val')
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

    miou_cal = runningScore(n_classes=19)
    with torch.no_grad():
        for step, data in enumerate(test_data_loader):
            print('{}/{}'.format(step, len(test_data_loader)))
            im_list, gt = data
            for i in range(len(im_list)):
                im_list[i] = im_list[i].cuda()
            gt = gt.squeeze().numpy()

            pred = net(im_list)
            out = torch.argmax(pred, dim=1)
            out = out.squeeze().cpu().numpy()
            miou_cal.update(gt, out)

#             if step == 10:
#                 break

        miou, iou = miou_cal.get_scores()
        miou_cal.reset()
        print('miou:{}'.format(miou))
        for i in range(len(iou)):
            print(iou[i])


if __name__ == '__main__':
    test()
