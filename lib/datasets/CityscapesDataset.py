import os
import cv2
import random
import numpy as np
import scipy.io as io

import torch
from torch.utils.data import Dataset


class CityscapesDataset(Dataset):
    def __init__(self, root_data_path, im_path, period, num_im=3, crop_size=None, resize_size=None, aug=True):
        self.dataset_dir = root_data_path
        self.im_path = im_path
        self.period = period
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.num_im = num_im
        self.aug = aug
        self.mean = np.array([123.675, 116.28, 103.53])
        self.mean = np.expand_dims(np.expand_dims(self.mean, axis=1), axis=1)
        self.std = np.array([58.395, 57.12, 57.375])
        self.std = np.expand_dims(np.expand_dims(self.std, axis=1), axis=1)

        self.get_list()

    def get_list(self):
        self.im_names = []
        self.gt_names = []

        file_path = os.path.join('/code/ST_Memory/data/Cityscapes/list', self.period + '.txt')
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            im_name, gt_name = line.split()

            im_id = int(im_name.split('_')[2])
            interval = 1

            name_list = []
            for i in range(self.num_im):
                name = im_name.replace(
                    '{:06d}_leftImg8bit.png'.format(im_id),
                    '{:06d}_leftImg8bit.png'.format(im_id + (i - self.num_im // 2) * interval),
                )
                name_list.append(name)

            self.im_names.append(name_list)
            self.gt_names.append(gt_name)

    def __len__(self):
        return len(self.gt_names)

    def __getitem__(self, idx):
        im_list = []
        for i in range(self.num_im):
            im = cv2.imread(os.path.join(self.im_path, 'leftImg8bit_sequence', self.period, self.im_names[idx][i]))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_list.append(im)
        gt = cv2.imread(os.path.join(self.dataset_dir, 'gtFine', self.period, self.gt_names[idx]), 0)
        
        if self.resize_size is not None:
            im_list, gt = self.resize(im_list, gt)
        
        if self.crop_size is not None:
            im_list, gt = self.crop(im_list, gt)
        
        if self.period == 'train' and self.aug:            
            im_list, gt = self.randomflip(im_list, gt)

        h, w = gt.shape
        im_list, gt = self.totentor(im_list, gt)

        return im_list, gt

    def resize(self, im_list, gt):
        resize_h, resize_w = self.resize_size
        for i in range(self.num_im):
            im_list[i] = cv2.resize(im_list[i], (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        return im_list, gt
    
    def crop(self, im_list, gt):
        h, w = gt.shape
        crop_h, crop_w = self.crop_size
        start_h = np.random.randint(h - crop_h)
        start_w = np.random.randint(w - crop_w)
        
        for i in range(self.num_im):
            im_list[i] = im_list[i][start_h:start_h+crop_h, start_w:start_w+crop_w, :]
        gt = gt[start_h:start_h+crop_h, start_w:start_w+crop_w]
        return im_list, gt

    def randomflip(self, im_list, gt):
        RANDOMFLIP = 0.5

        if np.random.rand() < RANDOMFLIP:
            for i in range(self.num_im):
                im_list[i] = np.flip(im_list[i], axis=1)
            gt = np.flip(gt, axis=1)

        return im_list, gt

    def totentor(self, im_list, gt):
        for i in range(self.num_im):
            im = im_list[i].transpose([2, 0, 1])
            im = (im - self.mean) / self.std
            im = torch.from_numpy(im.copy()).float()
            im_list[i] = im
        gt = torch.from_numpy(gt.copy()).long()

        return im_list, gt


class CityscapesDataset_Image(Dataset):
    def __init__(self, root_data_path, im_path, period, resize_size=None, aug=True):
        self.dataset_dir = root_data_path
        self.im_path = im_path
        self.period = period
        self.resize_size = resize_size
        self.aug = aug
        self.mean = np.array([123.675, 116.28, 103.53])
        self.mean = np.expand_dims(np.expand_dims(self.mean, axis=1), axis=1)
        self.std = np.array([58.395, 57.12, 57.375])
        self.std = np.expand_dims(np.expand_dims(self.std, axis=1), axis=1)

        self.get_list()

    def get_list(self):
        self.im_names = []
        self.gt_names = []

        file_path = os.path.join('/code/ST_Memory/data/Cityscapes/list', self.period + '.txt')
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            im_name, gt_name = line.split()
            self.im_names.append(im_name)
            self.gt_names.append(gt_name)

    def __len__(self):
        return len(self.gt_names)

    def __getitem__(self, idx):
        im = cv2.imread(os.path.join(self.im_path, 'leftImg8bit_sequence', self.period, self.im_names[idx]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(os.path.join(self.dataset_dir, 'gtFine', self.period, self.gt_names[idx]), 0)

        if self.period == 'train' and self.aug:
            if self.resize_size is not None:
                im, gt = self.resize(im, gt)
            im, gt = self.randomflip(im, gt)

        im, gt = self.totentor(im, gt)

        return im, gt

    def resize(self, im, gt):
        resize_h, resize_w = self.resize_size
        im = cv2.resize(im, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)

        return im, gt

    def randomflip(self, im, gt):
        RANDOMFLIP = 0.5

        if np.random.rand() < RANDOMFLIP:
            im = np.flip(im, axis=1)
            gt = np.flip(gt, axis=1)

        return im, gt

    def totentor(self, im, gt):
        im = im.transpose([2, 0, 1])
        im = (im - self.mean) / self.std
        im = torch.from_numpy(im.copy()).float()
        gt = torch.from_numpy(gt.copy()).long()

        return im, gt

class CityscapesDataset_Image_Aug(Dataset):
    def __init__(self, root_data_path, im_path, period, crop_size=None, aug=True):
        self.dataset_dir = root_data_path
        self.im_path = im_path
        self.period = period
        self.crop_size = crop_size
        self.aug = aug
        self.mean = np.array([123.675, 116.28, 103.53])
        self.mean = np.expand_dims(np.expand_dims(self.mean, axis=1), axis=1)
        self.std = np.array([58.395, 57.12, 57.375])
        self.std = np.expand_dims(np.expand_dims(self.std, axis=1), axis=1)

        self.get_list()

    def get_list(self):
        self.im_names = []
        self.gt_names = []

        file_path = os.path.join('/code/ST_Memory/data/Cityscapes/list', self.period + '.txt')
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            im_name, gt_name = line.split()
            self.im_names.append(im_name)
            self.gt_names.append(gt_name)

    def __len__(self):
        return len(self.gt_names)

    def __getitem__(self, idx):
        im = cv2.imread(os.path.join(self.im_path, 'leftImg8bit_sequence', self.period, self.im_names[idx]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(os.path.join(self.dataset_dir, 'gtFine', self.period, self.gt_names[idx]), 0)

        if self.period == 'train' and self.aug:
            if self.crop_size is not None:
                im, gt = self.randomcrop(im, gt)
            im, gt = self.randomflip(im, gt)

        im, gt = self.totentor(im, gt)

        return im, gt

    def randomcrop(self, im, gt):
        h, w = gt.shape
        crop_h, crop_w = self.crop_size
        h_start = random.randint(0, h - crop_h)
        w_start = random.randint(0, w - crop_w)

        im = im[h_start:h_start + crop_h, w_start:w_start + crop_w, :]
        gt = gt[h_start:h_start + crop_h, w_start:w_start + crop_w]

        return im, gt
    
    def randomflip(self, im, gt):
        RANDOMFLIP = 0.5

        if np.random.rand() < RANDOMFLIP:
            im = np.flip(im, axis=1)
            gt = np.flip(gt, axis=1)

        return im, gt

    def totentor(self, im, gt):
        im = im.transpose([2, 0, 1])
        im = (im - self.mean) / self.std
        im = torch.from_numpy(im.copy()).float()
        gt = torch.from_numpy(gt.copy()).long()

        return im, gt


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_path = '/data/bitahub/Cityscapes'
    seg_path = '/data/bitahub/Cityscapes'
    dataset = CityscapesDataset(data_path, seg_path, period='train', num_im=3)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        im_list, gt = data
        print('{}/{}'.format(i, len(dataloader)), len(im_list), im_list[0].shape, gt.shape)
