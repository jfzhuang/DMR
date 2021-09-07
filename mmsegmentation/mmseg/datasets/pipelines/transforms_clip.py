import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from ..builder import PIPELINES


@PIPELINES.register_module()
class Resize_Clip(object):
    def __init__(self, img_scale=None, multiscale_mode='range', ratio_range=None, keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h), self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        clip = results['clip']
        for i in range(len(clip)):
            if self.keep_ratio:
                clip[i], scale_factor = mmcv.imrescale(clip[i], results['scale'], return_scale=True)
                new_h, new_w = clip[i].shape[:2]
                h, w = clip[i].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                clip[i], w_scale, h_scale = mmcv.imresize(clip[i], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        results['clip'] = clip
        results['img_shape'] = clip[i].shape
        results['pad_shape'] = clip[i].shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(results[key], results['scale'], interpolation='nearest')
            results[key] = gt_seg

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(img_scale={self.img_scale}, '
            f'multiscale_mode={self.multiscale_mode}, '
            f'ratio_range={self.ratio_range}, '
            f'keep_ratio={self.keep_ratio})'
        )
        return repr_str


@PIPELINES.register_module()
class RandomCrop_Clip(object):
    def __init__(self, crop_size, cat_max_ratio=1.0, ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        clip = results['clip']
        crop_bbox = self.get_crop_bbox(clip[0])
        if self.cat_max_ratio < 1.0:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(clip[0])
        # crop the image
        for i in range(len(clip)):
            clip[i] = self.crop(clip[i], crop_bbox)
        img_shape = clip[0].shape
        results['clip'] = clip
        results['img_shape'] = img_shape

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class RandomFlip_Clip(object):
    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            clip = results['clip']
            for i in range(len(clip)):
                clip[i] = mmcv.imflip(clip[i], direction=results['flip_direction'])
            results['clip'] = clip

            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = mmcv.imflip(results[key], direction=results['flip_direction']).copy()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


@PIPELINES.register_module()
class PhotoMetricDistortion_Clip(object):
    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, clip, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        for i in range(len(clip)):
            clip[i] = clip[i].astype(np.float32) * alpha + beta
            clip[i] = np.clip(clip[i], 0, 255)
            clip[i] = clip[i].astype(np.uint8)
        return clip

    def brightness(self, clip):
        """Brightness distortion."""
        if random.randint(2):
            for i in range(len(clip)):
                clip[i] = self.convert(clip[i], beta=random.uniform(-self.brightness_delta, self.brightness_delta))
            return clip
        return clip

    def contrast(self, clip):
        """Contrast distortion."""
        if random.randint(2):
            for i in range(len(clip)):
                clip[i] = self.convert(clip[i], alpha=random.uniform(self.contrast_lower, self.contrast_upper))
            return clip
        return clip

    def saturation(self, clip):
        """Saturation distortion."""
        if random.randint(2):
            for i in range(len(clip)):
                clip[i] = mmcv.bgr2hsv(clip[i])
                clip[i][:, :, 1] = self.convert(
                    clip[i][:, :, 1], alpha=random.uniform(self.saturation_lower, self.saturation_upper)
                )
                clip[i] = mmcv.hsv2bgr(clip[i])
        return clip

    def hue(self, clip):
        """Hue distortion."""
        if random.randint(2):
            for i in range(len(clip)):
                clip[i] = mmcv.bgr2hsv(clip[i])
                clip[i][:, :, 0] = (
                    clip[i][:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)
                ) % 180
                clip[i] = mmcv.hsv2bgr(clip[i])
        return clip

    def __call__(self, results):
        clip = results['clip']
        # random brightness
        clip = self.brightness(clip)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            clip = self.contrast(clip)

        # random saturation
        clip = self.saturation(clip)

        # random hue
        clip = self.hue(clip)

        # random contrast
        if mode == 0:
            clip = self.contrast(clip)

        results['clip'] = clip
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(brightness_delta={self.brightness_delta}, '
            f'contrast_range=({self.contrast_lower}, '
            f'{self.contrast_upper}), '
            f'saturation_range=({self.saturation_lower}, '
            f'{self.saturation_upper}), '
            f'hue_delta={self.hue_delta})'
        )
        return repr_str


@PIPELINES.register_module()
class Normalize_Clip(object):
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        clip = results['clip']
        for i in range(len(clip)):
            clip[i] = mmcv.imnormalize(clip[i], self.mean, self.std, self.to_rgb)
        results['clip'] = clip
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' f'{self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class Pad_Clip(object):
    def __init__(self, size=None, size_divisor=None, pad_val=0, seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        clip = results['clip']
        for i in range(len(clip)):
            if self.size is not None:
                clip[i] = mmcv.impad(clip[i], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                clip[i] = mmcv.impad_to_multiple(clip[i], self.size_divisor, pad_val=self.pad_val)
        results['clip'] = clip
        results['pad_shape'] = clip[i].shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(results[key], shape=results['pad_shape'][:2], pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' f'pad_val={self.pad_val})'
        return repr_str
