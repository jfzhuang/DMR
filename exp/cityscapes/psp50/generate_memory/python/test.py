import os
import sys
import cv2
import random
import argparse
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.metrics import runningScore
from lib.datasets.CityscapesDataset import CityscapesDataset
from exp.cityscapes.psp50.STF.model.refine import SegRefine

def get_arguments():
    parser = argparse.ArgumentParser(description="Test")
    ###### general setting ######
    parser.add_argument("--data_path", type=str, help="path to the data")
    parser.add_argument("--im_path", type=str, help="path to the images")

    ###### inference setting ######
    parser.add_argument("--num_workers", type=int, help="num of cpus used")

    return parser.parse_args()


class Memory_Sampler(object):
    def __init__(self):
        self.values = []
        self.keys = []

        self.max_value = []
        self.min_value = []
        self.sample_num_max = []
        self.sample_num_min = []

        self.num_key_sample = 10
        self.num_value_sample = 10
        self.num_class = 19

        for i in range(self.num_class):
            self.values.append([])
            self.keys.append([])
            self.max_value.append(1.0)
            self.min_value.append(0.0)
            self.sample_num_max.append(0)
            self.sample_num_min.append(0)

    def update(self, sample):
        feat, score, class_idx = sample['feat'], sample['score'], sample['class_idx']

        flag_min = False
        flag_max = False
        if score < self.max_value[class_idx]:
            flag_min = True
        elif score > self.min_value[class_idx]:
            flag_max = True

        if flag_min:
            self.keys[class_idx].append(sample)
            self.keys[class_idx] = sorted(self.keys[class_idx], key=lambda x: x['score'])
            self.sample_num_min[class_idx] += 1
            if self.sample_num_min[class_idx] > self.num_key_sample:
                self.keys[class_idx].pop(-1)
                self.sample_num_min[class_idx] -= 1
            self.max_value[class_idx] = self.keys[class_idx][-1]['score']

            return True

        if flag_max:
            self.values[class_idx].append(sample)
            self.values[class_idx] = sorted(self.values[class_idx], key=lambda x: x['score'])
            self.sample_num_max[class_idx] += 1
            if self.sample_num_max[class_idx] > self.num_value_sample:
                self.values[class_idx].pop(0)
                self.sample_num_max[class_idx] -= 1
            self.min_value[class_idx] = self.values[class_idx][0]['score']

            return True

        return False

def generate_memory():
    args = get_arguments()
    print(args)

    net = SegRefine()
    net.eval().cuda()

    weight_path = '/data/jfzhuang/VSPW_480p/work_dirs_v2/ST_Memory/exp/cityscapes/psp50/STF/best.pth'
    weight = torch.load(weight_path, map_location='cpu')
    net.load_state_dict(weight, True)

    test_data = CityscapesDataset(args.data_path, args.im_path, 'train')
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_data_loader = iter(test_data_loader)

    sampler = Memory_Sampler()
    with torch.no_grad():
        for step in trange(len(test_data_loader)):
            im_list, gt = next(test_data_loader)
            for i in range(len(im_list)):
                im_list[i] = im_list[i].cuda()
            gt = gt.squeeze().numpy()
            gt = cv2.resize(gt, (256, 128), interpolation=cv2.INTER_NEAREST)

            feat, logits = net.forward_return_feat(im_list)
            score = F.softmax(logits, dim=1)
            out = torch.argmax(score, dim=1)
            score, _ = torch.max(score, dim=1)

            feat = feat.cpu().numpy()
            score = score.squeeze().cpu().numpy()
            mask = out.squeeze().cpu().numpy() == gt

            row, col = np.where(mask ==True)
            for i in range(len(row)):
                feat_sample = feat[:, :, row[i], col[i]]
                score_sample = score[row[i], col[i]]
                class_idx_sample = gt[row[i], col[i]]
                sample = {'score': score_sample, 'feat': feat_sample, 'class_idx': class_idx_sample}
                sampler.update(sample)

            output_0 = 'value num:'
            output_1 = 'key num:'
            for i in range(19):
                output_0 += ' {}'.format(sampler.sample_num_max[i])
                output_1 += ' {}'.format(sampler.sample_num_min[i])

            print(output_0)
            print(output_1)
            print()

#             if step == 5:
#                 break
        
        key = []
        for i in range(19):
            tmp = []
            for j in range(len(sampler.keys[i])):
                tmp.append(sampler.keys[i][j]['feat'])
            if len(tmp) < sampler.num_key_sample:
                tmp = tmp * (sampler.num_key_sample//len(tmp)+1)
                tmp = tmp[:sampler.num_key_sample]
            tmp = np.concatenate(tmp, axis=0)
            key.append(tmp)            
        key = np.concatenate(key, axis=0)
        np.save(
            '/code/ST_Memory/exp/cityscapes/psp50/generate_memory/output/key.npy', key)
        print('key:', key.shape)
        
        value = []
        for i in range(19):
            tmp = []
            for j in range(len(sampler.values[i])):
                tmp.append(sampler.values[i][j]['feat'])
            if len(tmp) < sampler.num_value_sample:
                tmp = tmp * (sampler.num_value_sample//len(tmp)+1)
                tmp = tmp[:sampler.num_value_sample]
            tmp = np.concatenate(tmp, axis=0)
            tmp = np.mean(tmp, axis=0, keepdims=True)
            value.append(tmp)            
        value = np.concatenate(value, axis=0)
        np.save(
            '/code/ST_Memory/exp/cityscapes/psp50/generate_memory/output/value.npy', value)
        print('value:', value.shape)
        
        
if __name__ == '__main__':
    generate_memory()

    