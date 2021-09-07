# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import cv2
import numpy as np
import torch
import random


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size, is_continuous=False, fix=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST
        self.fix = fix

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if self.output_size == (h, w):
            return sample

        if self.fix:
            h_rate = self.output_size[0] / h
            w_rate = self.output_size[1] / w
            min_rate = h_rate if h_rate < w_rate else w_rate
            new_h = h * min_rate
            new_w = w * min_rate
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv2.INTER_CUBIC)

        top = (self.output_size[0] - new_h) // 2
        bottom = self.output_size[0] - new_h - top
        left = (self.output_size[1] - new_w) // 2
        right = self.output_size[1] - new_w - left
        if self.fix:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        if 'segmentation' in sample.keys():
            segmentation = sample['segmentation']
            seg = cv2.resize(segmentation, dsize=(new_w, new_h), interpolation=self.seg_interpolation)
            if self.fix:
                seg = cv2.copyMakeBorder(seg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
            sample['segmentation'] = seg
        sample['image'] = img
        return sample


class Centerlize(object):
    def __init__(self, output_size, is_continuous=False):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if self.output_size == (h, w):
            return sample

        if isinstance(self.output_size, int):
            new_h = self.output_size
            new_w = self.output_size
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        top = (new_h-h) // 2
        bottom = new_h - h - top
        left = (new_w-w) // 2
        right = new_w - w - left
        img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if 'segmentation' in sample.keys():
            segmentation = sample['segmentation']
            seg = cv2.copyMakeBorder(segmentation, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
            sample['segmentation'] = seg
        sample['image'] = img

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        new_h = h if new_h >= h else new_h
        new_w = w if new_w >= w else new_w

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top:top + new_h, left:left + new_w]

        segmentation = segmentation[top:top + new_h, left:left + new_w]
        sample['image'] = image
        sample['segmentation'] = segmentation
        return sample


class RandomScaleCrop(object):
    """Randomly scale image and crop"""
    def __init__(self, size):
        self.crop_size = size

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        row, col, _ = image.shape

        scale_list = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        scale = scale_list[random.randint(0, 6)]
        resized_row, resized_col = int(row * scale), int(col * scale)

        img = cv2.resize(image, (resized_col, resized_row), interpolation=cv2.INTER_LINEAR)
        seg = cv2.resize(segmentation, (resized_col, resized_row), interpolation=cv2.INTER_NEAREST)

        crop_row = self.crop_size[0]
        crop_col = self.crop_size[1]
        pad_row = max(crop_row - resized_row, 0)
        pad_col = max(crop_col - resized_col, 0)

        img = np.pad(img, ((0, pad_row), (0, pad_col), (0, 0)), 'constant', constant_values=0.0)
        seg = np.pad(seg, ((0, pad_row), (0, pad_col)), 'constant', constant_values=255)

        row, col, _ = img.shape
        crop_x = random.randint(0, col - crop_col)
        crop_y = random.randint(0, row - crop_row)
        img = img[crop_y:crop_y + crop_row, crop_x:crop_x + crop_col, :]
        seg = seg[crop_y:crop_y + crop_row, crop_x:crop_x + crop_col]

        sample['image'] = img
        sample['segmentation'] = seg
        return sample


class RandomHSV(object):
    """Generate randomly the image in hsv space."""
    def __init__(self, h_r, s_r, v_r):
        self.h_r = h_r
        self.s_r = s_r
        self.v_r = v_r

    def __call__(self, sample):
        image = sample['image']
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h = hsv[:, :, 0].astype(np.int32)
        s = hsv[:, :, 1].astype(np.int32)
        v = hsv[:, :, 2].astype(np.int32)
        delta_h = np.random.randint(-self.h_r, self.h_r)
        delta_s = np.random.randint(-self.s_r, self.s_r)
        delta_v = np.random.randint(-self.v_r, self.v_r)
        h = (h+delta_h) % 180
        s = s + delta_s
        s[s > 255] = 255
        s[s < 0] = 0
        v = v + delta_v
        v[v > 255] = 255
        v[v < 0] = 0
        hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
        sample['image'] = image
        return sample


class RandomFlip(object):
    """Randomly flip image"""
    def __init__(self, threshold):
        self.flip_t = threshold

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        if np.random.rand() < self.flip_t:
            image_flip = np.flip(image, axis=1)
            segmentation_flip = np.flip(segmentation, axis=1)
            sample['image'] = image_flip
            sample['segmentation'] = segmentation_flip
        return sample


class RandomRotation(object):
    """Randomly rotate image"""
    def __init__(self, angle_r, is_continuous=False):
        self.angle_r = angle_r
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        row, col, _ = image.shape
        rand_angle = np.random.randint(-self.angle_r, self.angle_r) if self.angle_r != 0 else 0
        m = cv2.getRotationMatrix2D(center=(col / 2, row / 2), angle=rand_angle, scale=1)
        new_image = cv2.warpAffine(image, m, (col, row), flags=cv2.INTER_CUBIC, borderValue=0)
        new_segmentation = cv2.warpAffine(segmentation, m, (col, row), flags=self.seg_interpolation, borderValue=0)
        sample['image'] = new_image
        sample['segmentation'] = new_segmentation
        return sample


class RandomScale(object):
    """Randomly scale image"""
    def __init__(self, scale_r, is_continuous=False):
        self.scale_r = scale_r
        self.seg_interpolation = cv2.INTER_CUBIC if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        row, col, _ = image.shape
        rand_scale = np.random.rand() * (self.scale_r - 1 / self.scale_r) + 1 / self.scale_r
        img = cv2.resize(image, None, fx=rand_scale, fy=rand_scale, interpolation=cv2.INTER_CUBIC)
        seg = cv2.resize(segmentation, None, fx=rand_scale, fy=rand_scale, interpolation=self.seg_interpolation)
        sample['image'] = img
        sample['segmentation'] = seg
        return sample


class Multiscale(object):
    def __init__(self, rate_list):
        self.rate_list = rate_list

    def __call__(self, sample):
        image = sample['image']
        row, col, _ = image.shape
        image_multiscale = []
        for rate in self.rate_list:
            rescaled_image = cv2.resize(image, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
            sample['image_%f' % rate] = rescaled_image
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        key_list = sample.keys()
        for key in key_list:
            if 'image' in key:
                image = sample[key]
                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W
                image = image.transpose((2, 0, 1))
                sample[key] = torch.from_numpy(image.astype(np.float32) / 255.0)
                #sample[key] = torch.from_numpy(image.astype(np.float32)/128.0-1.0)
            elif 'segmentation' == key:
                segmentation = sample['segmentation']
                sample['segmentation'] = torch.from_numpy(segmentation.astype(np.int64))
            elif 'segmentation_onehot' == key:
                onehot = sample['segmentation_onehot'].transpose((2, 0, 1))
                sample['segmentation_onehot'] = torch.from_numpy(onehot.astype(np.int64))
            elif 'mask' == key:
                mask = sample['mask']
                sample['mask'] = torch.from_numpy(mask.astype(np.int64))
        return sample


def onehot(label, num):
    m = label
    one_hot = np.eye(num)[m]
    return one_hot


def Apollo_gt_convert(label):
    class_table = [[17, 0], [33, 1], [34, 2], [35, 3], [36, 4], [37, 5], [38, 6], [39, 7], [40, 8], [49, 9], [50, 10],
                   [65, 11], [66, 12], [67, 13], [81, 14], [82, 15], [83, 16], [84, 17], [85, 255], [86, 255], [97, 18],
                   [98, 255], [99, 255], [100, 19], [113, 20], [0, 255]]

    class_id = [
        0, 1, 17, 33, 161, 34, 162, 35, 163, 36, 164, 37, 165, 38, 166, 39, 167, 40, 168, 49, 50, 65, 66, 67, 81, 82,
        83, 84, 85, 86, 97, 98, 99, 100, 113, 255
    ]
    train_id = [
        255, 255, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 255,
        255, 255, 21, 255
    ]

    class_table = [[0, 255], [1, 255], [17, 0], [33, 1], [161, 1], [34, 2], [162, 2], [35, 3], [163, 3], [36, 4],
                   [164, 4], [37, 5], [165, 5], [38, 6], [166, 6], [39, 7], [167, 7], [40, 8], [168, 8], [49, 9],
                   [50, 10], [65, 11], [66, 12], [67, 13], [81, 14], [82, 15], [83, 16], [84, 17], [85, 18], [86, 19],
                   [97, 20], [98, 255], [99, 255], [100, 255], [113, 21]]

    out = np.zeros([label.shape[0], label.shape[1]], dtype=np.uint8)
    for i in range(len(class_table)):
        mask = label == class_table[i][0]
        out += mask.astype(np.uint8) * class_table[i][1]
    return out
